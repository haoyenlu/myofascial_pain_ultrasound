"""
5-fold CV training pipeline, updated with the fixes from the training
diagnostics discussion:

1. Frozen-layer bug fixed: `param.requires_grad = False` (was the typo
   `require_grad`, which silently did nothing -- the whole backbone was
   being fine-tuned instead of just the head).
2. Weight decay excludes 1D params (norm gains, biases) -- only conv/linear
   weight matrices get decayed. Also lowered the decay value itself
   (1e-2 -> 1e-4) in config.py.
3. Gradient clipping added after loss.backward(), before optimizer.step().
4. `drop_last=True` on the train loader, to remove the small trailing-batch
   effect that was producing grad-norm spikes (we saw norms up to 15.8 on
   partial batches, vs. a normal range of ~1.5-6).
5. LR schedule: linear warmup for `warmup_epochs`, then cosine anneal to
   `min_lr` -- unchanged in shape from before, just now paired with fixes
   1-3 so the warmup doesn't blow up an accidentally-unfrozen backbone.
6. Checkpoint selection uses the EMA-smoothed combined score
   (ema_val_auc - ema_val_loss), not a single epoch's raw AUC or loss --
   a spike from one noisy validation pass no longer gets picked as "best".
7. No fixed holdout. All subjects go through 5-fold CV (via build_folds.py).
   Each outer fold's clip predictions are saved, then pooled across all 5
   folds at the end for out-of-fold (OOF) evaluation --
   this is the primary evaluation, not any single fold or a small holdout.
8. BCEWithLogitsLoss uses pos_weight from the fold's training class balance.
9. After checkpoint selection, maximize Youden J on unaugmented training
   clip predictions and apply that fold's cutoff to validation clips.
   Fold and pooled CSVs include youden_threshold and youden_prediction.
   This uses in-sample training scores, not inner out-of-fold calibration;
   checkpoint selection uses outer validation metrics as described below.
10. Train one model per fold on the other four folds. The outer validation
    fold selects the checkpoint, so its reported metrics are also used for
    model selection rather than being an independent test evaluation.

This still assumes the same surrounding project modules as before
(models.resnet, datasets.video_loader/imgResize/img2tensor, vidaug) --
only the training loop, data split, and evaluation logic changed.
"""
import os
import copy
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vidaug.augmentors as va
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config, ResnetConfig, VDEConfig
from datasets import img2tensor, imgResize, video_loader
from cohort import healthy_latent_cohort
from reproducibility import seed_training, fold_seed, seed_worker

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class VideoSample:
    label: int
    video: str
    subject_id: str
    intervention: str
    video_id: str
    frame_indices: list
    side: str
    sex: int = 0
    age: float = 0.0
    ppt: tuple = (0.0, 0.0)


class VideoDataset(Dataset):
    def __init__(self, videos, labels, subjects, interventions, sides,
                 frame_size, sampler, augmentation=None, repeat_chn=False,
                 use_demo=False, sex_features=None, age_features=None,
                 ppt_features=None):
        assert len(videos) == len(labels) == len(subjects)
        self.videos = videos
        self.labels = labels
        self.subjects = subjects
        self.interventions = interventions
        self.sides = sides
        self.frame_size = frame_size
        self.sampler = sampler
        self.augmentation = augmentation
        self.repeat_chn = repeat_chn
        self.use_demo = use_demo
        self.sex_features = sex_features
        self.age_features = age_features
        self.ppt_features = ppt_features
        if self.use_demo:
            assert sex_features is not None and age_features is not None and ppt_features is not None
            assert len(videos) == len(sex_features) == len(age_features) == len(ppt_features)
        self.frame_cache = {}
        self.data = self._prepare()

    def _prepare(self):
        data = []
        for i, (v, l) in enumerate(zip(self.videos, self.labels)):
            frames = self._get_frames(v)
            for s in self.sampler(len(frames)):
                data.append(VideoSample(
                    label=l, video=v, subject_id=self.subjects[i],
                    intervention=self.interventions[i], frame_indices=s,
                    side=self.sides[i], video_id=Path(v).stem,
                    sex=int(self.sex_features[i]) if self.use_demo else 0,
                    age=float(self.age_features[i]) if self.use_demo else 0.0,
                    ppt=tuple(float(x) for x in self.ppt_features[i]) if self.use_demo else (0.0, 0.0),
                ))
        return data

    def _get_frames(self, video_path):
        if video_path not in self.frame_cache:
            self.frame_cache[video_path] = video_loader(video_path)
        return self.frame_cache[video_path]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        frames = self._get_frames(s.video)
        clip = [frames[i] for i in s.frame_indices]
        clip = self.augmentation(clip) if self.augmentation is not None else clip
        clip = [imgResize(img, target_size=self.frame_size) for img in clip]
        clip = img2tensor(np.array(clip), self.repeat_chn)
        item = {
            "clip": clip,
            "label": torch.Tensor((s.label,)).float(),
            "subject": s.subject_id,
            "intervention": s.intervention,
            "side": s.side,
            "video_id": s.video_id,
        }
        if self.use_demo:
            item["demo"] = {
                "sex": torch.tensor([s.sex], dtype=torch.long),
                "age": torch.tensor([s.age], dtype=torch.float32),
                "ppt": torch.tensor(s.ppt, dtype=torch.float32),
            }
        return item


def get_dataloader(config: Config, df, is_train=True):
    if is_train and config.augmentation:
        sometimes = lambda aug: va.Sometimes(config.augmentation_prob, aug)
        augmentation = va.Sequential([
            sometimes(va.HorizontalFlip()),
            sometimes(va.VerticalFlip()),
            sometimes(va.RandomRotate(config.rotation_degree)),
            sometimes(va.RandomTranslate(config.translation_degree, y=config.translation_degree)),
            sometimes(va.GaussianBlur(config.gaussian_blur_sigma)),
            sometimes(va.RandomShear(config.shear_degree, config.shear_degree)),
        ])
    else:
        augmentation = None

    sampler = va.SlidingWindowSampler(clip_len=config.clip_len, step=config.step_size)
    dataset = VideoDataset(
        videos=df["Video"].to_list(),
        labels=df["label"].to_list(),
        subjects=df["Study ID"].to_list(),
        interventions=df["Intervention"].to_list(),
        sides=df["Side"].to_list(),
        frame_size=config.frame_size,
        sampler=sampler,
        repeat_chn=config.repeat_chn,
        augmentation=augmentation,
        use_demo=getattr(config, "use_demo", False),
        sex_features=df["sex_label"].to_list() if getattr(config, "use_demo", False) else None,
        age_features=df["Age_STD"].to_list() if getattr(config, "use_demo", False) else None,
        ppt_features=df[["PPT_PRE_STD", "PPT_POST_STD"]].to_numpy().tolist()
        if getattr(config, "use_demo", False) else None,
    )
    # drop_last=True on TRAIN only -- removes the small trailing batch that
    # was causing grad-norm spikes. Val keeps every sample (drop_last=False).
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=is_train,
                       drop_last=is_train, worker_init_fn=seed_worker,
                       generator=torch.Generator().manual_seed(getattr(config, 'active_fold_seed', config.seed)))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DataWrapper(nn.Module):
    def __init__(self, model, image_key: str = "clip"):
        super().__init__()
        self.model = model
        self.image_key = image_key
        self.device = torch.device("cuda")

    def forward(self, data: dict):
        return self.model(data[self.image_key].to(self.device))


def get_model(config, fold_id=None, device=torch.device("cuda")):
    """Build the model for the current fold.

    `fold_id` is needed for VDE, which loads a per-fold pretrained
    diffusion checkpoint (`config.diffusion_fold_checkpoint[f"fold-{fold_id}"]`).
    It's ignored for the resnet variants.
    """
    if config.model == "VDE":
        return get_vde_model(config, fold_id, device)
    elif config.model in ("resnet50", "resnet101"):
        return get_resnet_model(config, device)
    else:
        raise ValueError(f"Unknown config.model={config.model!r}")


def get_resnet_model(config: ResnetConfig, device=torch.device("cuda")):
    from models.resnet import resnet50, resnet101

    resnet_fn    = resnet101 if config.model == "resnet101" else resnet50
    resnet_model = resnet_fn(**config.resnet_param)
    state_dict = torch.load(config.resnet_pretrained_path, map_location=device)["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if not k.startswith("fc.")}
    resnet_model.load_state_dict(state_dict, strict=False)

    # Fixed: was `param.require_grad = False` (typo -> silent no-op).
    for name, param in resnet_model.named_parameters():
        if name.startswith(config.freeze_layers):
            param.requires_grad = False

    model = DataWrapper(resnet_model)
    model.to(device)
    return model


def get_vde_model(config: VDEConfig, fold_id=None, device=torch.device("cuda")):
    """Build the VDE (video-diffusion-encoder + 3D ResNet head) model.

    Unlike the resnet path, VDE is called directly on the batch dict
    (`model(data)`), not through `DataWrapper` -- it consumes both the
    `clip` tensor and the `demo` dict internally (matching how it's used
    in main.py). No `DataWrapper` device-move is needed here since VDE
    handles that itself.
    """
    from models.vde import VDE

    fold_key = f"fold-{fold_id}"
    diffusion_ckpt = config.diffusion_fold_checkpoint[fold_key]

    frozen_source = getattr(config, "frozen_vde_checkpoint_dir", None)
    model = VDE(
        diffusion_ckpt=diffusion_ckpt,
        unet_config=config.unet_param,
        diffusion_config=config.diffusion_param,
        encoder_config=config.encoder_param,
        resnet_config=config.resnet_head_param,
        use_demo=False if frozen_source else config.use_demo,
        sex_embedding_dim=config.demo_param["sex_embedding"],
        age_embedding_dim=config.demo_param["age_embedding"],
        ppt_embedding_dim=config.demo_param["ppt_embedding"],
        num_ppt=config.demo_param["num_ppt"],
        ppt_only=getattr(config, "ppt_only", False),
        intervention_matched_ppt=getattr(config, "intervention_matched_ppt", False),
    )
    model.to(device)
    print(f"Using timestep={config.encoder_param['timestep']}  "
          f"block_to_use={config.encoder_param['block_to_use']}")

    if frozen_source:
        from models.vde import FrozenVDEWithDemographics
        state = torch.load(Path(frozen_source) / fold_key / "best_combined_model.pth",
                           map_location=device, weights_only=True)["state_dict"]
        model.load_state_dict(state, strict=True)
        model = FrozenVDEWithDemographics(model, config.demo_param["sex_embedding"],
                    config.demo_param["age_embedding"], config.demo_param["ppt_embedding"]).to(device)
        print(f"Loaded frozen image checkpoint: {frozen_source}/{fold_key}; training demographic branch only")
        return model

    # Optional partial freezing (e.g. keep the pretrained diffusion U-Net
    # encoder fixed and only fine-tune the resnet head / demo branch).
    # `config.freeze_layers` defaults to `()` -- i.e. full end-to-end
    # fine-tuning, matching the original VDE training setup in main.py.
    # Same fixed-typo pattern as the resnet path: `requires_grad`, not
    # `require_grad`.
    for name, param in model.named_parameters():
        if name.startswith(config.freeze_layers):
            param.requires_grad = False

    return model




def get_param_groups(model, weight_decay, exclude_norm_and_bias=True):
    """1D params (biases, norm weights) skip weight decay; everything else gets it."""
    if not exclude_norm_and_bias:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_lr_scheduler(optimizer, config: Config):
    """Linear warmup for `warmup_epochs`, then cosine anneal to min_lr."""
    if config.warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=config.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.min_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
    )


# ---------------------------------------------------------------------------
# EMA-smoothed metric tracking for checkpoint selection
# ---------------------------------------------------------------------------

class EMATracker:
    """Zero-initialized EMA: every epoch, including epoch 0, gets weight alpha.

    No bias correction is applied, so early scores are shrunk toward zero.
    """
    def __init__(self, alpha):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.ema_val_loss = 0.0
        self.ema_val_auc = 0.0
        self.best_combined = -float("inf")

    def update(self, val_loss, val_auc):
        a = self.alpha
        self.ema_val_loss = a * val_loss + (1 - a) * self.ema_val_loss
        self.ema_val_auc = a * val_auc + (1 - a) * self.ema_val_auc
        combined = self.ema_val_auc - self.ema_val_loss
        return self.ema_val_loss, self.ema_val_auc, combined

    def is_best(self, combined_score):
        if combined_score > self.best_combined:
            self.best_combined = combined_score
            return True
        return False


# ---------------------------------------------------------------------------
# Train / validate one fold
# ---------------------------------------------------------------------------

def youden_threshold(labels, probabilities):
    """Fit on training clips; ties select the highest observed threshold."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not labels.size:
        raise ValueError("Youden fitting requires nonempty, matching 1D arrays")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Youden fitting requires both training classes (0 and 1)")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Youden fitting requires finite probabilities in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    # Exclude the all-negative sentinel across sklearn versions (inf or max+1).
    valid = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= probabilities.max()))
    best = valid[np.argmax((tpr - fpr)[valid])]
    return float(thresholds[best])


def fit_training_youden(model, config, train_df, fold_id):
    """Evaluate the selected model on all unaugmented training clips."""
    train_eval_dl = get_dataloader(config, train_df, is_train=False)
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(train_eval_dl, desc=f"fold {fold_id} train Youden"):
            probabilities.extend(torch.sigmoid(model(data)).cpu().numpy().reshape(-1).tolist())
            labels.extend(data["label"].cpu().numpy().reshape(-1).tolist())
    threshold = youden_threshold(labels, probabilities)
    print(f"[fold {fold_id}] training-clip Youden threshold={threshold:.6f} (n={len(labels)})")
    return threshold


def apply_youden_threshold(predictions, threshold, probability_column="probability"):
    """Retain fold-specific training cutoffs when pooling validation rows."""
    predictions = predictions.copy()
    predictions["youden_threshold"] = threshold
    predictions["youden_prediction"] = (predictions[probability_column] >= threshold).astype(int)
    return predictions


def split_inner_subjects(frame, seed=42):
    """80:20 subject split, stratified by each subject's label profile."""
    if frame["Study ID"].isna().any():
        raise ValueError("Subject IDs must not be missing")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Outer training data must contain binary labels 0 and 1")
    profiles = frame.groupby("Study ID")["label"].agg(
        lambda values: ",".join(str(int(v)) for v in sorted(set(values))))
    train_ids, val_ids = train_test_split(
        profiles.index.to_numpy(), test_size=0.2, random_state=seed,
        stratify=profiles.to_numpy())
    train = frame[frame["Study ID"].isin(train_ids)].copy().reset_index(drop=True)
    val = frame[frame["Study ID"].isin(val_ids)].copy().reset_index(drop=True)
    for name, partition in (("inner training", train), ("inner validation", val)):
        if set(partition.label.unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both labels 0 and 1; refusing to train")
    return train, val


def train_fold(config: ResnetConfig, train_df, val_df, fold_id, patience=None,
               outer_test_df=None, return_inner_predictions=False):
    """
    patience: if given, stop training this fold early once the EMA-smoothed
      combined score (ema_val_auc - ema_val_loss) hasn't improved for this
      many CONSECUTIVE epochs. Selection and patience start at epoch 0,
      including learning-rate warmup epochs.
      None (default) = no early stopping, train the full config.epochs
      budget, unchanged from before this parameter existed.
    """
    selection_start_epoch = 0
    if not 0 <= selection_start_epoch < config.epochs:
        raise ValueError("epochs must be positive to select a checkpoint")
    print(f"[fold {fold_id}] checkpoint selection starts at epoch {selection_start_epoch} (zero-based)")
    config.active_fold_seed = seed_training(fold_seed(config.seed, fold_id))
    print(f'[fold {fold_id}] training seed={config.active_fold_seed}')
    device = torch.device("cuda")
    model = get_model(config, fold_id, device)

    train_dl = get_dataloader(config, train_df, is_train=True)
    val_dl = get_dataloader(config, val_df, is_train=False)
    print(f"[fold {fold_id}] train clips: {len(train_dl.dataset)}  val clips: {len(val_dl.dataset)}")

    n_pos = sum(sample.label for sample in train_dl.dataset.data)
    n_neg = len(train_dl.dataset) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    param_groups = get_param_groups(model, config.weight_decay, config.no_decay_norm_and_bias)
    optimizer = optim.AdamW(param_groups, lr=config.lr)
    scheduler = get_lr_scheduler(optimizer, config)

    checkpoint_path = os.path.join(config.checkpoint_dir, f"fold-{fold_id}")
    os.makedirs(checkpoint_path, exist_ok=True)

    tracker = EMATracker(alpha=config.ema_loss_alpha)
    history = dict(train_loss=[], val_loss=[], auc=[], lr=[], grad_norm=[],
                    ema_val_loss=[], ema_val_auc=[], combined_score=[])
    best_val_probs, best_val_labels, best_val_subjects = None, None, None
    # Keep predictions from the best checkpoint, including warmup epochs.
    best_track = None
    epochs_since_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss, train_steps = 0.0, 0
        grad_norms = []
        for data in tqdm(train_dl, desc=f"fold {fold_id} epoch {epoch} train"):
            labels = data["label"].to(device).float()
            B = labels.shape[0]

            outputs = model(data)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            grad_norms.append(gnorm.item())

            optimizer.step()
            train_loss += loss.item() * B
            train_steps += B

        if train_steps == 0:
            raise ValueError("Training loader produced no samples; check batch size and dataset")
        train_loss /= train_steps
        scheduler.step()

        model.eval()
        val_loss, val_steps = 0.0, 0
        val_track = dict(val_probs=[], val_labels=[], val_subjects=[],val_video_id=[], val_interventions = [], val_side = [])
        with torch.no_grad():
            for data in tqdm(val_dl, desc=f"fold {fold_id} epoch {epoch} val"):
                labels = data["label"].to(device).float()
                B = labels.shape[0]
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * B
                val_steps += B

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_track["val_probs"].extend(probs.reshape(-1).tolist())
                val_track["val_labels"].extend(labels.cpu().numpy().reshape(-1).tolist())
                val_track['val_subjects'].extend(data["subject"])
                val_track['val_video_id'].extend(data['video_id'])
                val_track['val_interventions'].extend(data['intervention'])
                val_track['val_side'].extend(data['side'])

        val_loss /= val_steps
        val_auc = roc_auc_score(val_track['val_labels'], val_track['val_probs'])
        if epoch == selection_start_epoch:
            # Initialize selection tracking from the very first epoch.
            tracker = EMATracker(alpha=config.ema_loss_alpha)
        ema_val_loss, ema_val_auc, combined_score = tracker.update(val_loss, val_auc)
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"[fold {fold_id}] epoch {epoch}: train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} auc={val_auc:.3f} "
              f"ema_auc={ema_val_auc:.3f} combined={combined_score:.3f} lr={curr_lr:.2e} "
              f"grad_norm(mean/max)={np.mean(grad_norms):.2f}/{np.max(grad_norms):.2f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["auc"].append(val_auc)
        history["lr"].append(curr_lr)
        history["grad_norm"].append(float(np.max(grad_norms)))
        history["ema_val_loss"].append(ema_val_loss)
        history["ema_val_auc"].append(ema_val_auc)
        history["combined_score"].append(combined_score)

        selection_eligible = epoch >= selection_start_epoch
        is_best_combined = selection_eligible and tracker.is_best(combined_score)
        if is_best_combined:
            torch.save({"state_dict": model.state_dict(), "seed": config.active_fold_seed},
                       os.path.join(checkpoint_path, "best_combined_model.pth"))
            # The supplied validation split selects the checkpoint.
            best_track = val_track
            epochs_since_improvement = 0
            print(f"[fold {fold_id}] new best combined_score={combined_score:.4f} at epoch {epoch}")
        elif selection_eligible:
            epochs_since_improvement += 1

        if selection_eligible and patience is not None and epochs_since_improvement >= patience:
            print(f"[fold {fold_id}] EARLY STOPPING at epoch {epoch} -- combined_score hasn't "
                  f"improved for {patience} consecutive epochs "
                  f"(best={tracker.best_combined:.4f})")
            break

    pd.DataFrame(history).to_csv(os.path.join(checkpoint_path, "training_history.csv"), index=False)

    if return_inner_predictions:
        inner = pd.DataFrame(best_track).rename(columns={
            "val_probs": "probability", "val_labels": "label",
            "val_subjects": "subject", "val_video_id": "video_id",
            "val_interventions": "intervention", "val_side": "side",
        })
        inner.to_csv(os.path.join(checkpoint_path, "inner_validation_predictions.csv"), index=False)
        return inner

    if outer_test_df is not None:
        # Release training state before loading the selected model for outer OOF.
        del model, optimizer, scheduler, train_dl, val_dl
        return run_inference_fold(config, train_df, outer_test_df, fold_id,
                                  output_name="oof_val_result.csv")

    # Match the selected checkpoint to the saved validation predictions.
    model.load_state_dict(torch.load(
        os.path.join(checkpoint_path, "best_combined_model.pth"),
        map_location=device,
    )["state_dict"])
    del train_dl  # Release the video cache before loading training clips again.
    threshold = fit_training_youden(model, config, train_df, fold_id)
    oof_df = apply_youden_threshold(pd.DataFrame(best_track), threshold, "val_probs")
    oof_df['fold_id'] = fold_id
    oof_df.to_csv(os.path.join(checkpoint_path, "oof_val_result.csv"), index=False)
    return oof_df


def run_inference_fold(config, train_df, val_df, fold_id,
                       checkpoint_name="best_combined_model.pth",
                       output_name="oof_val_result_inference.csv", threshold=None):
    """
    Inference-only counterpart to train_fold: load the already-trained
    checkpoint for this fold and run a single forward pass over its
    validation split -- no optimizer, no scheduler, no backward pass, no
    training loop over epochs.

    The caller fits age/PPT scalers on train_df. Here the training split
    also supplies clip probabilities for fitting the Youden threshold.
    Validation labels never participate in threshold fitting.

    Returns an oof_df with columns (subject, label, probability, video_id,
    intervention, side, fold) -- matching what `evaluate_pooled_oof`
    expects, plus youden_threshold and youden_prediction. Training output
    uses legacy column names which evaluate_pooled_oof normalizes.
    """
    device = torch.device("cuda")
    model = get_model(config, fold_id, device)

    checkpoint_path = os.path.join(config.checkpoint_dir, f"fold-{fold_id}")
    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_file} -- check config.checkpoint_dir "
            f"and that fold {fold_id} finished training."
        )
    state_dict = torch.load(checkpoint_file, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[fold {fold_id}] loaded checkpoint: {checkpoint_file}")

    val_dl = get_dataloader(config, val_df, is_train=False)
    print(f"[fold {fold_id}] val clips: {len(val_dl.dataset)}")

    n_pos = sum(train_df["label"])
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_track = dict(probability=[], label=[], subject=[], video_id=[], intervention=[], side=[])
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for data in tqdm(val_dl, desc=f"fold {fold_id} inference"):
            labels = data["label"].to(device).float()
            B = labels.shape[0]
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * B
            val_steps += B

            probs = torch.sigmoid(outputs).cpu().numpy()
            val_track["probability"].extend(probs.reshape(-1).tolist())
            val_track["label"].extend(labels.cpu().numpy().reshape(-1).tolist())
            val_track["subject"].extend(data["subject"])
            val_track["video_id"].extend(data["video_id"])
            val_track["intervention"].extend(data["intervention"])
            val_track["side"].extend(data["side"])

    val_loss /= val_steps
    val_auc = roc_auc_score(val_track["label"], val_track["probability"])
    print(f"[fold {fold_id}] inference val_loss={val_loss:.3f} val_auc={val_auc:.3f}")

    oof_df = pd.DataFrame(val_track)
    oof_df["fold"] = fold_id
    if threshold is None:
        threshold = fit_training_youden(model, config, train_df, fold_id)
    oof_df = apply_youden_threshold(oof_df, threshold)

    os.makedirs(checkpoint_path, exist_ok=True)
    oof_df.to_csv(os.path.join(checkpoint_path, output_name), index=False)
    return oof_df


# ---------------------------------------------------------------------------
# Evaluation: pooled out-of-fold
# ---------------------------------------------------------------------------

def evaluate_pooled_oof(oof_dfs, out_dir, out_name="pooled_oof.csv"):
    pooled = pd.concat(oof_dfs, ignore_index=True)
    pooled = pooled.rename(columns={
        "val_probs": "probability",
        "val_labels": "label",
        "val_subjects": "subject",
        "val_video_id": "video_id",
        "val_interventions": "intervention",
        "val_side": "side",
        "fold_id": "fold",
    })
    pooled.to_csv(os.path.join(out_dir, out_name), index=False)

    sample_auc = roc_auc_score(pooled["label"], pooled["probability"])

    # Opposite sides can have different diagnoses; never take the first
    # side's label as the label of the entire bilateral subject.
    subj_level = pooled.groupby(["subject", "side", "intervention"]).agg(
        mean_prob=("probability", "mean"), label=("label", "first")
    )
    subject_auc = roc_auc_score(subj_level["label"], subj_level["mean_prob"])

    print(f"\nPooled OOF sample-level AUC: {sample_auc:.4f} (n={len(pooled)})")
    print(f"Pooled OOF subject/side/intervention AUC: {subject_auc:.4f} (n={len(subj_level)})")

    fpr, tpr, _ = roc_curve(pooled["label"], pooled["probability"])
    fpr_s, tpr_s, _ = roc_curve(subj_level["label"], subj_level["mean_prob"])
    np.savez(os.path.join(out_dir, os.path.splitext(out_name)[0] + "_roc.npz"),
             fpr=fpr, tpr=tpr, sample_auc=sample_auc,
             fpr_subj=fpr_s, tpr_subj=tpr_s, subject_auc=subject_auc)
    return sample_auc, subject_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inner_cv_partitions(df, outer_fold):
    """Use the four existing non-outer folds as inner validation folds."""
    if df["Study ID"].isna().any() or df.groupby("Study ID")["cv_fold"].nunique().gt(1).any():
        raise ValueError("Each subject must belong to exactly one fold")
    fold_ids = sorted(df["cv_fold"].unique())
    if len(fold_ids) != 5 or outer_fold not in fold_ids:
        raise ValueError("Four-model inner CV requires five existing folds")
    for inner_fold in fold_ids:
        if inner_fold == outer_fold:
            continue
        train = df[~df.cv_fold.isin([outer_fold, inner_fold])].copy().reset_index(drop=True)
        val = df[df.cv_fold == inner_fold].copy().reset_index(drop=True)
        outer = df[df.cv_fold == outer_fold].copy().reset_index(drop=True)
        for name, partition in (("inner train", train), ("inner validation", val), ("outer", outer)):
            if partition.label.isna().any() or set(partition.label.unique()) != {0, 1}:
                raise ValueError(f"{name} must contain both binary labels")
        yield int(inner_fold), train, val, outer


def ensemble_outer_predictions(predictions, threshold, outer_fold):
    """Average aligned clip probabilities, retaining all four member scores."""
    first = next(iter(predictions.values()))
    metadata = ["subject", "video_id", "intervention", "side", "label"]
    result = first[metadata].copy().reset_index(drop=True)
    score_columns = []
    for inner_fold, frame in predictions.items():
        if not frame[metadata].reset_index(drop=True).equals(result[metadata]):
            raise ValueError("Inner models produced misaligned outer clip predictions")
        column = f"probability_inner_fold_{inner_fold}"
        result[column] = frame["probability"].to_numpy()
        score_columns.append(column)
    if len(score_columns) != 4:
        raise ValueError("The outer ensemble requires all four inner models")
    result["probability"] = result[score_columns].mean(axis=1)
    result["fold"] = outer_fold
    return apply_youden_threshold(result, threshold)


def run_outer_inner_cv(config, df, outer_fold, patience=None, inference=False,
                       checkpoint_name="best_combined_model.pth", output_name="oof_val_result.csv"):
    """Select four inner models, then evaluate their mean on the outer fold.

    The shared cutoff is fitted to pooled inner validation scores. These
    scores also select checkpoints; this is not an independent calibration set.
    VDE members share the pretrained diffusion backbone for the outer fold.
    """
    outer_dir = Path(config.checkpoint_dir) / f"fold-{outer_fold}"
    outer_dir.mkdir(parents=True, exist_ok=True)
    member_predictions = {}
    inner_predictions = []
    for inner_fold, train, val, outer in inner_cv_partitions(df, outer_fold):
        member_config = copy.deepcopy(config)
        # train_fold/get_model keep the OUTER id for the pretrained backbone.
        member_config.checkpoint_dir = str(outer_dir / f"inner-{inner_fold}")
        member_config.seed = config.seed + inner_fold * config.n_fold
        age_scaler = StandardScaler().fit(train[["Age"]])
        ppt_scaler = StandardScaler().fit(train[["PPT_PRE_AVG", "PPT_POST_AVG"]])
        for partition in (train, val, outer):
            partition["Age_STD"] = age_scaler.transform(partition[["Age"]])
            partition[["PPT_PRE_STD", "PPT_POST_STD"]] = ppt_scaler.transform(
                partition[["PPT_PRE_AVG", "PPT_POST_AVG"]])
        print(f"Outer fold {outer_fold}, inner validation fold {inner_fold}: "
              f"train subjects={train['Study ID'].nunique()}, val subjects={val['Study ID'].nunique()}")
        if inference:
            inner = run_inference_fold(member_config, train, val, outer_fold,
                checkpoint_name, "inner_validation_predictions_inference.csv", threshold=0.5)
        else:
            inner = train_fold(member_config, train, val, outer_fold, patience=patience,
                               return_inner_predictions=True)
        inner["inner_fold"] = inner_fold
        inner_predictions.append(inner)
        member_predictions[inner_fold] = run_inference_fold(
            member_config, train, outer, outer_fold, checkpoint_name,
            "outer_predictions.csv", threshold=youden_threshold(inner.label, inner.probability))
    pooled_inner = pd.concat(inner_predictions, ignore_index=True)
    threshold = youden_threshold(pooled_inner.label, pooled_inner.probability)
    result = ensemble_outer_predictions(member_predictions, threshold, outer_fold)
    result.to_csv(outer_dir / output_name, index=False)
    print(f"[outer fold {outer_fold}] four-model ensemble, inner OOF Youden={threshold:.6f}")
    return result


def config_to_dict(config):
    """
    Accepts either a config class or an instance. `vars(SomeClass)` only
    returns that class's OWN namespace, not attributes inherited from a
    base class (e.g. VDEConfig doesn't define `epochs` itself -- it's
    inherited from Config) -- so this walks the full MRO, base class
    first, to collect every inherited default. For an instance, any
    attributes set directly on the instance (e.g. by apply_vde_overrides)
    are merged in last so they correctly take precedence over the class
    defaults they override.
    """
    cls = config if isinstance(config, type) else type(config)
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass))
    if not isinstance(config, type):
        merged.update(vars(config))
    return {k: v for k, v in merged.items() if not k.startswith("__") and not callable(v)}

def apply_vde_overrides(config, timestep=None, block_to_use=None, eval_seed=None,
                        checkpoint_tag=None, checkpoint_root=None):
    """
    Override VDEConfig's diffusion timestep and/or which U-Net block the
    encoder reads features from, via function arguments/CLI flags instead
    of editing config.py directly. No-op (with a warning) for non-VDE
    configs, since neither parameter means anything there.

    `config.timestep` and `config.encoder_param["timestep"]` are two
    separate things in this codebase -- the former is effectively just
    for display (only `encoder_param["timestep"]` is actually read by
    DiffusionEncoder), but both are kept in sync here so logs/config.yaml
    dumps stay consistent with what's actually used.

    checkpoint_dir is always regenerated from whichever timestep/
    block_to_use end up in effect (CLI override, or the config's own
    default if neither is passed) -- so the checkpoint directory name
    always reflects what this run actually used, not just what happens
    to be written in config.py at the time.
    """
    if not hasattr(config, "encoder_param"):
        if timestep is not None or block_to_use is not None:
            print("WARNING: --timestep/--block_to_use only apply to VDE configs; ignored here.")
        return config

    # Class-level dictionaries are shared between instances; copy before
    # applying run-specific values so one inference call cannot leak into another.
    config.encoder_param = dict(config.encoder_param)
    if timestep is not None:
        config.timestep = timestep
        config.encoder_param["timestep"] = timestep
    if block_to_use is not None:
        config.encoder_param["block_to_use"] = block_to_use
    if eval_seed is not None:
        config.encoder_param["eval_seed"] = eval_seed

    effective_timestep = config.encoder_param["timestep"]
    effective_block = config.encoder_param["block_to_use"]
    if 'block_to_use' in config.resnet_head_param:
        config.resnet_head_param = dict(config.resnet_head_param)
        config.resnet_head_param['block_to_use'] = effective_block
        if 'pool_size' in config.encoder_param:
            config.resnet_head_param['input_dim'] = config.encoder_param['pool_size']
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    if checkpoint_root:
        config.checkpoint_dir = str(Path(checkpoint_root) / f"block{effective_block}")
    else:
        config.checkpoint_dir = (
            f"./checkpoints/vde_cv_t{effective_timestep}_block{effective_block}{tag_suffix}"
        )
    print(f"VDE config in effect: timestep={effective_timestep}  block_to_use={effective_block}  "
          f"-> checkpoint_dir={config.checkpoint_dir}")
    return config


def load_saved_vde_model_config(config, eval_seed=None):
    """Restore architecture/data settings saved beside a trained checkpoint.

    This keeps inference and visualization compatible when the live VDEConfig
    has changed since training. Run-location fields and the requested eval seed
    remain controlled by the current invocation.
    """
    config_path = Path(config.checkpoint_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing saved checkpoint config: {config_path}")
    with config_path.open("r") as stream:
        saved = yaml.load(stream, Loader=yaml.FullLoader)

    for key in (
        "data_csv", "repeat_chn", "clip_len", "frame_size", "use_demo",
        "step_size", "diffusion_fold_checkpoint", "timestep",
        "unet_param", "diffusion_param", "resnet_head_param", "demo_param",
        "inner_split_seed", "inner_validation_fraction",
        "inner_cv_folds", "seed", "n_fold",
        "frozen_vde_checkpoint_dir", "ppt_only", "intervention_matched_ppt",
    ):
        if key in saved:
            setattr(config, key, saved[key])
    config.encoder_param = dict(saved["encoder_param"])
    if eval_seed is not None:
        config.encoder_param["eval_seed"] = eval_seed
    print(f"Loaded saved model configuration from {config_path}")
    return config


def apply_vde_profile(config, profile=None):
    """Apply a named, reproducible VDE architecture profile."""
    if profile is None:
        return config
    if profile != "compact_head_v2":
        raise ValueError(f"Unknown VDE profile: {profile}")
    config.encoder_param = dict(config.encoder_param)
    config.encoder_param["hidden_chn"] = 16
    config.resnet_head_param = {
        "input_dim": config.encoder_param.get("pool_size", 16),
        "planes_3d_blocks": [16, 32],
        "pool_size": 2,
        "hidden_dim": 64,
        "dropout_prob": 0.3,
        "stride": 2,
        "block_to_use": 13
    }
    config.vde_profile = profile
    print(f"Applied VDE profile: {profile}")
    return config


CONFIG_REGISTRY = {
    "resnet50": ResnetConfig,
    "resnet101": ResnetConfig,
    "VDE": VDEConfig,
}
MODEL_NAME = "VDE"


def main(folds=None, timestep=None, block_to_use=None, epochs=None, patience=None,
         checkpoint_tag=None, vde_profile=None, clip_len=None, step_size=None,
         use_demo=None, checkpoint_root=None, seed=None, encoder_pool_size=None,
         inner_split_seed=42, frozen_vde_checkpoint_dir=None, warmup_epochs=None, ppt_only=False, intervention_matched_ppt=False):
    """
    folds: which fold(s) to train, overriding config.train_folds if given.
      - None (default): use config.train_folds as-is (unchanged behavior).
      - "all": train every fold, 1..config.n_fold.
      - a list of ints, e.g. [2, 4]: train exactly those folds.
    timestep, block_to_use: [VDE only] override config.py's values for
      the diffusion timestep / U-Net block the encoder reads from.
      checkpoint_dir is auto-named from whichever values end up in
      effect -- see apply_vde_overrides.
    epochs: override config.py's epochs for this run (e.g. a smaller
      budget for a quick screening sweep, vs. config.py's full budget
      for a final run). None (default) = use config.epochs as-is.
    patience: stop each fold's training early once the EMA-smoothed combined
      score hasn't improved for this many consecutive epochs -- see train_fold's
      docstring. None (default) = no early stopping, train the full
      epoch budget (config.epochs, or the override above) for every fold.
    """
    ConfigClass = CONFIG_REGISTRY[MODEL_NAME]
    config = ConfigClass()
    config.model = MODEL_NAME
    # Legacy inner_split_seed argument is retained for caller compatibility.
    if frozen_vde_checkpoint_dir is not None:
        config.checkpoint_dir = str(Path(frozen_vde_checkpoint_dir).resolve())
        config = load_saved_vde_model_config(config)
        config.frozen_vde_checkpoint_dir = config.checkpoint_dir
        config.use_demo = True
    if warmup_epochs is not None:
        config.warmup_epochs = warmup_epochs
    config.intervention_matched_ppt = intervention_matched_ppt
    if intervention_matched_ppt:
        if frozen_vde_checkpoint_dir:
            raise ValueError("Intervention-matched PPT requires the standard VDE model")
        use_demo = True
    config.ppt_only = ppt_only
    if ppt_only:
        if frozen_vde_checkpoint_dir:
            raise ValueError("PPT-only mode is not supported with the frozen demographic wrapper")
        use_demo = True
    config.inner_cv_folds = 0
    if seed is not None:
        config.seed = seed
    seed_training(config.seed)
    config = apply_vde_profile(config, vde_profile)
    if encoder_pool_size is not None:
        if encoder_pool_size <= 0:
            raise ValueError('encoder_pool_size must be positive')
        config.encoder_param = dict(config.encoder_param, pool_size=encoder_pool_size)
        config.resnet_head_param = dict(config.resnet_head_param, input_dim=encoder_pool_size)
        checkpoint_tag = f'{checkpoint_tag}_pool{encoder_pool_size}' if checkpoint_tag else f'pool{encoder_pool_size}'
        if checkpoint_root:
            checkpoint_root = str(Path(checkpoint_root) / f'pool{encoder_pool_size}')
    config = apply_vde_overrides(
        config, timestep=timestep, block_to_use=block_to_use,
        checkpoint_tag=checkpoint_tag, checkpoint_root=checkpoint_root,
    )
    if clip_len is not None:
        print(f"Overriding config.clip_len: {config.clip_len} -> {clip_len}")
        config.clip_len = clip_len
    if step_size is not None:
        print(f"Overriding config.step_size: {config.step_size} -> {step_size}")
        config.step_size = step_size
    if use_demo is not None:
        print(f"Overriding config.use_demo: {config.use_demo} -> {use_demo}")
        config.use_demo = use_demo
    if epochs is not None:
        print(f"Overriding config.epochs: {config.epochs} -> {epochs}")
        config.epochs = epochs
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)

    if folds is None:
        folds_to_run = list(config.train_folds)
    elif folds == "all":
        folds_to_run = list(range(1, config.n_fold + 1))
    else:
        folds_to_run = list(folds)

    invalid = [f for f in folds_to_run if f not in range(1, config.n_fold + 1)]
    if invalid:
        raise ValueError(f"Invalid fold id(s) {invalid} -- must be in 1..{config.n_fold}")
    print(f"Training folds: {folds_to_run}")
    if patience is not None:
        print(f"Early stopping enabled: combined_score patience={patience} epochs")

    df = healthy_latent_cohort(pd.read_csv(config.data_csv), config.n_fold)
    df["sex_label"] = np.where(df["Sex"] == "M", 0, 1)

    oof_dfs = []
    for fold_id in range(1, config.n_fold + 1):
        if fold_id not in folds_to_run: continue
        
        train_df = df[df["cv_fold"] != fold_id].copy().reset_index(drop=True)
        val_df = df[df["cv_fold"] == fold_id].copy().reset_index(drop=True)
        if set(train_df["Study ID"]) & set(val_df["Study ID"]):
            raise ValueError("Subject leakage between training and validation")
        age_scaler = StandardScaler().fit(train_df[["Age"]])
        ppt_scaler = StandardScaler().fit(train_df[["PPT_PRE_AVG", "PPT_POST_AVG"]])
        for partition in (train_df, val_df):
            partition["Age_STD"] = age_scaler.transform(partition[["Age"]])
            partition[["PPT_PRE_STD", "PPT_POST_STD"]] = ppt_scaler.transform(
                partition[["PPT_PRE_AVG", "PPT_POST_AVG"]])
        print(f"\n=== Fold {fold_id}/{config.n_fold}: outer validation selects checkpoint ===")
        print(f"train subjects={train_df['Study ID'].nunique()}, "
              f"validation subjects={val_df['Study ID'].nunique()}")
        t0 = time.time()
        oof_df = train_fold(config, train_df, val_df, fold_id, patience=patience)
        oof_dfs.append(oof_df)
        print(f"[outer fold {fold_id}] done in {timedelta(seconds=int(time.time() - t0))}")

    evaluate_pooled_oof(oof_dfs, config.checkpoint_dir)


def run_inference(folds=None, checkpoint_name="best_combined_model.pth",
                   pooled_out_name=None, timestep=None, block_to_use=None,
                   eval_seed=None, checkpoint_tag=None, checkpoint_dir=None):
    """
    Inference-only counterpart to main(): instead of training, load each
    fold's already-trained checkpoint and evaluate it on that fold's
    validation split once. Same fold-selection semantics as main()'s
    `folds` argument (None -> config.train_folds, "all" -> every fold,
    or an explicit list).

    timestep, block_to_use: [VDE only] must match whatever the checkpoint
      you're loading was actually trained with -- these change which
      checkpoint_dir is read from (see apply_vde_overrides), and change
      the model architecture/feature source itself, so a mismatch here
      will load weights into the wrong shape or silently evaluate with
      different features than the checkpoint was trained on.

    Folds without a saved checkpoint are skipped with a warning rather
    than raising -- useful if you're only interested in whichever folds
    happen to have finished training so far.
    """
    ConfigClass = CONFIG_REGISTRY[MODEL_NAME]
    config = ConfigClass()
    config.model = MODEL_NAME
    config = apply_vde_overrides(config, timestep=timestep, block_to_use=block_to_use,
                                 eval_seed=eval_seed, checkpoint_tag=checkpoint_tag)
    if checkpoint_dir is not None:
        config.checkpoint_dir = str(Path(checkpoint_dir).resolve())
    config = load_saved_vde_model_config(config, eval_seed=eval_seed)
    if eval_seed is None:
        eval_seed = config.encoder_param.get("eval_seed", 0)
    fold_out_name = f"oof_val_result_inference_seed{eval_seed}.csv"
    if pooled_out_name is None:
        pooled_out_name = f"pooled_oof_inference_seed{eval_seed}.csv"
    print(f"Inference eval_seed={eval_seed}; outputs use suffix seed{eval_seed}")

    if folds is None:
        folds_to_run = list(config.train_folds)
    elif folds == "all":
        folds_to_run = list(range(1, config.n_fold + 1))
    else:
        folds_to_run = list(folds)

    invalid = [f for f in folds_to_run if f not in range(1, config.n_fold + 1)]
    if invalid:
        raise ValueError(f"Invalid fold id(s) {invalid} -- must be in 1..{config.n_fold}")
    print(f"Running inference on folds: {folds_to_run}")

    df = healthy_latent_cohort(pd.read_csv(config.data_csv), config.n_fold)
    df["sex_label"] = np.where(df["Sex"] == "M", 0, 1)

    oof_dfs = []
    for fold_id in range(1, config.n_fold + 1):
        if fold_id not in folds_to_run:
            continue

        if getattr(config, "inner_cv_folds", None) == 4:
            oof_dfs.append(run_outer_inner_cv(
                config, df, fold_id, inference=True,
                checkpoint_name=checkpoint_name, output_name=fold_out_name))
            continue

        train_df = df[df["cv_fold"] != fold_id].reset_index(drop=True)
        val_df = df[df["cv_fold"] == fold_id].reset_index(drop=True)

        # New checkpoints use an inner split; legacy configs retain their original split.
        if getattr(config, "inner_validation_fraction", None) is not None:
            if config.inner_validation_fraction != 0.2:
                raise ValueError("Only an 80:20 inner split is supported")
            train_df, _ = split_inner_subjects(train_df, seed=config.inner_split_seed + fold_id)

        # Same scaler-fitting as main() -- must be fit on this fold's TRAIN
        # subjects only, never on val, or the demo features leak fold info.
        age_scaler = StandardScaler().fit(train_df[["Age"]])
        ppt_scaler = StandardScaler().fit(train_df[["PPT_PRE_AVG", "PPT_POST_AVG"]])
        for d in (train_df, val_df):
            d["Age_STD"] = age_scaler.transform(d[["Age"]])
            d[["PPT_PRE_STD", "PPT_POST_STD"]] = ppt_scaler.transform(d[["PPT_PRE_AVG", "PPT_POST_AVG"]])

        print(f"\n=== Fold {fold_id}/{config.n_fold} (inference only) ===")
        try:
            oof_df = run_inference_fold(config, train_df, val_df, fold_id,
                                         checkpoint_name, fold_out_name)
        except FileNotFoundError as e:
            print(f"[fold {fold_id}] SKIPPED -- {e}")
            continue
        oof_dfs.append(oof_df)

    if not oof_dfs:
        print("No folds produced results (no checkpoints found) -- nothing to pool.")
        return None

    pd.concat(oof_dfs, ignore_index=True)[["fold", "youden_threshold"]].drop_duplicates().to_csv(
        Path(config.checkpoint_dir) / f"fold_youden_thresholds_inference_seed{eval_seed}.csv",
        index=False,
    )
    return evaluate_pooled_oof(oof_dfs, config.checkpoint_dir, out_name=pooled_out_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run inference for the 5-fold CV pipeline (resnet/VDE)")
    parser.add_argument('--model', choices=sorted(CONFIG_REGISTRY), default=MODEL_NAME,
                        help='Classifier to train/evaluate (default: %(default)s).')
    parser.add_argument('--frozen_vde_checkpoint_dir', default=None,
                        help='No-demo VDE checkpoint family to freeze; train only added demographics.')
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--intervention_matched_ppt', action='store_true',
                        help='Use only the PPT matching each clip intervention.')
    parser.add_argument('--ppt_only', action='store_true', help='Use pre/post PPT only; omit age/sex branches.')
    parser.add_argument('--inner_split_seed', type=int, default=42,
                        help='Legacy compatibility option; unused for outer-fold selection.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Training base seed; fold i uses seed+i-1 (default config.seed=42).')
    parser.add_argument("--infer", action="store_true",
                         help="Inference-only mode: load each fold's saved checkpoint and "
                              "evaluate it once, instead of training from scratch.")
    parser.add_argument("--folds", nargs="+", default=None,
                         help="Which fold(s) to process. E.g. '--folds 2 4' does only folds 2 "
                              "and 4; '--folds all' does every fold (1..n_fold). If omitted, "
                              "falls back to config.train_folds.")
    parser.add_argument("--checkpoint_name", default="best_combined_model.pth",
                         help="[--infer mode only] checkpoint filename to load from each fold's "
                              "checkpoint directory.")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="[--infer only] exact checkpoint directory containing config.yaml and fold directories.")
    parser.add_argument("--timestep", type=int, default=None,
                         help="[VDE only] override config.py's diffusion timestep used for "
                              "feature extraction. checkpoint_dir is auto-named to include this "
                              "value -- e.g. --timestep 100 --block_to_use 13 writes to "
                              "./checkpoints/vde_cv_t100_block13. In --infer mode, this must "
                              "match the value the checkpoint you're loading was trained with.")
    parser.add_argument("--block_to_use", type=int, default=None,
                         help="[VDE only] override config.py's U-Net block (0-17) the encoder "
                              "reads features from. Same checkpoint_dir auto-naming and "
                              "--infer-mode caveat as --timestep.")
    parser.add_argument("--eval_seed", type=int, default=None,
                         help="[VDE --infer only] deterministic diffusion noise seed. Fold and "
                              "pooled inference outputs are named with this seed.")
    parser.add_argument("--checkpoint_tag", default=None,
                         help="Optional suffix for a distinct checkpoint family, e.g. "
                              "'compact_head_v2' creates "
                              "vde_cv_t50_block15_compact_head_v2.")
    parser.add_argument("--checkpoint_root", default=None,
                        help="Optional parent directory for a structured block sweep. "
                             "Each run writes to CHECKPOINT_ROOT/blockN.")
    parser.add_argument("--vde_profile", choices=["compact_head_v2"], default=None,
                         help="Apply a frozen, named VDE architecture profile instead of "
                              "depending on later edits to config.py.")
    parser.add_argument("--clip_len", type=int, default=None,
                        help="[train mode only] override the number of frames per clip.")
    parser.add_argument("--step_size", type=int, default=None,
                        help="[train mode only] override the temporal sampling step size.")
    parser.add_argument('--encoder_pool_size', type=int, default=None,
                        help='Encoder spatial pool size, e.g. 32 or 64; also sets head input_dim.')
    parser.add_argument("--use_demo", action="store_true", default=None,
                        help="[VDE train mode only] enable sex, age, and PPT demographic fusion.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="[train mode only] override config.py's epochs for this run -- "
                              "e.g. a smaller budget for a quick screening sweep, vs. the full "
                              "budget for a final run. Default: use config.epochs as-is.")
    parser.add_argument("--patience", type=int, default=None,
                         help="[train mode only] stop each fold's training early once the "
                              "EMA combined score hasn't improved for this many consecutive epochs. "
                              "Default: None (no early stopping, train the full epoch budget).")
    args = parser.parse_args()
    MODEL_NAME = args.model

    if args.folds is None:
        folds = None
    elif len(args.folds) == 1 and args.folds[0].lower() == "all":
        folds = "all"
    else:
        folds = [int(f) for f in args.folds]

    if args.infer:
        run_inference(folds=folds, checkpoint_name=args.checkpoint_name,
                       timestep=args.timestep, block_to_use=args.block_to_use,
                       eval_seed=args.eval_seed, checkpoint_tag=args.checkpoint_tag,
                       checkpoint_dir=args.checkpoint_dir)
    else:
        main(folds=folds, timestep=args.timestep, block_to_use=args.block_to_use,
             epochs=args.epochs, patience=args.patience,
             checkpoint_tag=args.checkpoint_tag, vde_profile=args.vde_profile,
             clip_len=args.clip_len, step_size=args.step_size,
             use_demo=args.use_demo, checkpoint_root=args.checkpoint_root, seed=args.seed,
             encoder_pool_size=args.encoder_pool_size, inner_split_seed=args.inner_split_seed,
             frozen_vde_checkpoint_dir=args.frozen_vde_checkpoint_dir,
             warmup_epochs=args.warmup_epochs, ppt_only=args.ppt_only,
             intervention_matched_ppt=args.intervention_matched_ppt)
