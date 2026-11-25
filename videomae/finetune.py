r"""
Finetune VideoMAE (ViT-based) for MTrP classification, starting from each
fold's pretrained encoder (pretrain_videomae.py --fold <n>).

This does NOT reuse train_cv.py's training loop -- VideoMAEForVideoClassification
expects (batch, num_frames, channels, height, width) tensors, the opposite
frame/channel order from the ResNet50 pipeline's (batch, channels, frames,
height, width), and has a completely different module structure (transformer
blocks, not named conv layers) for freezing. Reusing train_cv.py here would
mean forcing a mismatched tensor format through it, which is worse than a
short, self-contained loop.

What IS carried over, deliberately, from the rest of this project's pipeline:
- EMA-smoothed combined_score checkpoint selection (not raw per-epoch AUC)
- Gradient clipping, drop_last=True on the train loader
- pos_weight computed from each fold's actual class balance
- No fixed holdout -- 5-fold CV, pooled OOF evaluation across all folds
- Fold-restricted throughout: fold k's classifier is finetuned starting
  from fold k's OWN pretrained encoder (never trained/pretrained on fold
  k's held-out subjects at any stage)

Usage:
    python finetune_videomae.py                          # finetune all 5 folds
    python finetune_videomae.py --infer                   # inference only, all 5 folds
    python finetune_videomae.py --infer --folds 2 4        # inference only, folds 2 and 4
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import vidaug.augmentors as va
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import VMAEConfig
from datasets import imgResize, video_loader
from videomae.pretrain import IMAGENET_MEAN, IMAGENET_STD, frame_to_normalized_tensor


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class VideoMAEClassificationDataset(Dataset):
    """Same sliding-window precompute pattern as everywhere else in this
    project; returns pixel_values in VideoMAE's (T, C, H, W) order plus the
    label/subject/video_id metadata needed for OOF pooling."""
    def __init__(self, df, num_frames, step_size, image_size, is_train, config):
        self.image_size = image_size
        self.frame_cache = {}
        sampler = va.SlidingWindowSampler(clip_len=num_frames, step=step_size)

        self.transform = None
        if is_train:
            self.transform = va.Sequential([
                va.Sometimes(0.6, va.HorizontalFlip()),
                va.Sometimes(0.6, va.RandomRotate(config.rotation_degree if hasattr(config, "rotation_degree") else 15)),
            ])

        self.samples = []
        for _, row in df.iterrows():
            path = row["Video"]
            frames = self._get_frames(path)
            for window in sampler(len(frames)):
                self.samples.append((path, window, row["label"], row["Study ID"], row.get("video_id", path),
                                      row.get("Intervention", ""), row.get("Side", "")))

    def _get_frames(self, path):
        if path not in self.frame_cache:
            self.frame_cache[path] = video_loader(path)
        return self.frame_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame_indices, label, subject, video_id, intervention, side = self.samples[idx]
        frames = self._get_frames(path)
        clip = [frames[i] for i in frame_indices]
        if self.transform is not None:
            clip = self.transform(clip)
        pixel_values = frame_to_normalized_tensor(clip, self.image_size)
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.float32),
            "subject": subject,
            "video_id": video_id,
            "intervention": intervention,
            "side": side,
        }


def get_dataloader(df, num_frames, step_size, image_size, is_train, batch_size, config):
    dataset = VideoMAEClassificationDataset(df, num_frames, step_size, image_size, is_train, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train)


# ---------------------------------------------------------------------------
# Model + freezing
# ---------------------------------------------------------------------------

def build_model(pretrained_model_name, fold_encoder_path, num_frozen_blocks, linear_probe_only, device):
    from transformers import VideoMAEForVideoClassification, VideoMAEModel

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name, num_labels=1
    )
    # overwrite the (Kinetics-pretrained) encoder with THIS fold's
    # pretrained-then-domain-adapted encoder
    fold_encoder = VideoMAEModel.from_pretrained(fold_encoder_path)
    model.videomae.load_state_dict(fold_encoder.state_dict())

    # freezing -- see VMAEConfig docstring for the reasoning
    for p in model.videomae.embeddings.parameters():
        p.requires_grad = False

    blocks = model.videomae.encoder.layer
    if linear_probe_only:
        for blk in blocks:
            for p in blk.parameters():
                p.requires_grad = False
    else:
        for i, blk in enumerate(blocks):
            trainable = i >= num_frozen_blocks
            for p in blk.parameters():
                p.requires_grad = trainable

    # the head (fc_norm + classifier) is always trainable -- it's freshly
    # initialized, never part of the MAE pretraining checkpoint, so freezing
    # it would leave it stuck at random init
    if model.fc_norm is not None:
        for p in model.fc_norm.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# EMA tracker (same pattern as train_cv.py)
# ---------------------------------------------------------------------------

class EMATracker:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema_val_loss = None
        self.ema_val_auc = None
        self.best_combined = -float("inf")

    def update(self, val_loss, val_auc):
        if self.ema_val_loss is None:
            self.ema_val_loss, self.ema_val_auc = val_loss, val_auc
        else:
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

def train_fold(fold_id, config, train_df, val_df, pretrained_encoder_dir, init_mode,
                out_dir, num_frozen_blocks, linear_probe_only):
    from transformers import VideoMAEForVideoClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_encoder_path = os.path.join(pretrained_encoder_dir, init_mode, f"fold-{fold_id}", "encoder")
    if not os.path.exists(fold_encoder_path):
        raise FileNotFoundError(f"No pretrained encoder at {fold_encoder_path} -- "
                                 f"run `python pretrain_videomae.py --fold {fold_id}"
                                 f"{' --from_scratch' if init_mode == 'scratch' else ''}` first.")

    model = build_model(config.pretrained_model_name, fold_encoder_path,
                         num_frozen_blocks, linear_probe_only, device)
    num_frames = model.config.num_frames
    image_size = model.config.image_size

    train_dl = get_dataloader(train_df, num_frames, config.step_size, image_size,
                               is_train=True, batch_size=config.ft_batch_size, config=config)
    val_dl = get_dataloader(val_df, num_frames, config.step_size, image_size,
                             is_train=False, batch_size=config.ft_batch_size, config=config)
    print(f"[fold {fold_id}] train clips: {len(train_dl.dataset)}  val clips: {len(val_dl.dataset)}")

    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.ft_lr, weight_decay=config.ft_weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=config.ft_warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.ft_epochs - config.ft_warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                       milestones=[config.ft_warmup_epochs])

    tracker = EMATracker(alpha=config.ema_loss_alpha)
    history = dict(train_loss=[], val_loss=[], auc=[], lr=[], grad_norm=[],
                    ema_val_loss=[], ema_val_auc=[], combined_score=[])
    best_val_probs, best_val_labels, best_val_subjects, best_val_video_ids = None, None, None, None
    best_val_inter, best_val_side = None, None
    for epoch in range(config.ft_epochs):
        model.train()
        train_loss, train_steps = 0.0, 0
        grad_norms = []
        for batch in tqdm(train_dl, desc=f"fold {fold_id} epoch {epoch} train"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            B = labels.shape[0]

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
            grad_norms.append(gnorm.item())
            optimizer.step()

            train_loss += loss.item() * B
            train_steps += B

        train_loss /= train_steps
        scheduler.step()

        model.eval()
        val_loss, val_steps = 0.0, 0
        val_probs, val_labels, val_subjects, val_video_ids = [], [], [], []
        val_inter, val_side = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"fold {fold_id} epoch {epoch} val"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                B = labels.shape[0]

                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                val_loss += loss.item() * B
                val_steps += B

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs.reshape(-1).tolist())
                val_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
                val_subjects.extend(batch["subject"])
                val_video_ids.extend(batch["video_id"])
                val_inter.extend(batch['intervention'])
                val_side.extend(batch['side'])


        val_loss /= val_steps
        val_auc = roc_auc_score(val_labels, val_probs)
        ema_val_loss, ema_val_auc, combined_score = tracker.update(val_loss, val_auc)
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"[fold {fold_id}] epoch {epoch}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"auc={val_auc:.3f} ema_auc={ema_val_auc:.3f} combined={combined_score:.3f} "
              f"lr={curr_lr:.2e} grad_norm(mean/max)={np.mean(grad_norms):.2f}/{np.max(grad_norms):.2f}")

        for k, v in [("train_loss", train_loss), ("val_loss", val_loss), ("auc", val_auc),
                     ("lr", curr_lr), ("grad_norm", float(np.max(grad_norms))),
                     ("ema_val_loss", ema_val_loss), ("ema_val_auc", ema_val_auc),
                     ("combined_score", combined_score)]:
            history[k].append(v)

        if tracker.is_best(combined_score):
            checkpoint_path = os.path.join(out_dir, f"fold-{fold_id}")
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(os.path.join(checkpoint_path, "best_combined_model"))
            best_val_probs, best_val_labels = val_probs, val_labels
            best_val_subjects, best_val_video_ids = val_subjects, val_video_ids
            best_val_inter, best_val_side = val_inter, val_side
            print(f"[fold {fold_id}] new best combined_score={combined_score:.4f} at epoch {epoch}")

    checkpoint_path = os.path.join(out_dir, f"fold-{fold_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(checkpoint_path, "training_history.csv"), index=False)

    oof_df = pd.DataFrame({
        "subject": best_val_subjects, "label": best_val_labels,
        "probability": best_val_probs, "video_id": best_val_video_ids, 
        "intervention": best_val_inter, "side": best_val_side,
        "fold": fold_id,
    })
    oof_df.to_csv(os.path.join(checkpoint_path, "oof_val_result.csv"), index=False)
    return oof_df


# ---------------------------------------------------------------------------
# Inference-only: load an already-finetuned checkpoint, evaluate once
# ---------------------------------------------------------------------------

def run_inference_fold(fold_id, config, val_df, out_dir, checkpoint_name="best_combined_model", train_df=None):
    """
    Inference-only counterpart to train_fold: load the already-finetuned
    checkpoint for this fold (saved via `model.save_pretrained(...)` in
    train_fold) and run a single forward pass over its validation split --
    no optimizer, no scheduler, no backward pass, no epoch loop.

    Unlike train_fold, this does NOT need `pretrained_encoder_dir`/
    `init_mode`/`num_frozen_blocks`/`linear_probe_only` -- those are only
    needed to build a *fresh* model to start training from. Here we load
    the fully-finetuned weights directly via
    `VideoMAEForVideoClassification.from_pretrained(checkpoint_path)`,
    which already has the right architecture and trained weights baked in.

    `train_df`, if given, is used only to compute pos_weight so the
    reported val_loss is on the same scale as training-time logs -- it
    has no effect on the predictions themselves. Pass None to skip loss
    reporting entirely and just get probabilities/AUC.

    Returns an oof_df with columns (subject, label, probability, video_id,
    intervention, side, fold).
    """
    from transformers import VideoMAEForVideoClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(out_dir, f"fold-{fold_id}", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- check `out_dir` and that fold {fold_id} "
            f"finished finetuning."
        )
    model = VideoMAEForVideoClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"[fold {fold_id}] loaded checkpoint: {checkpoint_path}")

    num_frames = model.config.num_frames
    image_size = model.config.image_size
    val_dl = get_dataloader(val_df, num_frames, config.step_size, image_size,
                             is_train=False, batch_size=config.ft_batch_size, config=config)
    print(f"[fold {fold_id}] val clips: {len(val_dl.dataset)}")

    criterion = None
    if train_df is not None:
        n_pos = train_df["label"].sum()
        n_neg = len(train_df) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_track = dict(probability=[], label=[], subject=[], video_id=[], intervention=[], side=[])
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"fold {fold_id} inference"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            B = labels.shape[0]

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            if criterion is not None:
                val_loss += criterion(logits, labels).item() * B
                val_steps += B

            probs = torch.sigmoid(logits).cpu().numpy()
            val_track["probability"].extend(probs.reshape(-1).tolist())
            val_track["label"].extend(labels.cpu().numpy().reshape(-1).tolist())
            val_track["subject"].extend(batch["subject"])
            val_track["video_id"].extend(batch["video_id"])
            val_track["intervention"].extend(batch["intervention"])
            val_track["side"].extend(batch["side"])

    val_auc = roc_auc_score(val_track["label"], val_track["probability"])
    if criterion is not None:
        print(f"[fold {fold_id}] inference val_loss={val_loss / val_steps:.3f}  val_auc={val_auc:.3f}")
    else:
        print(f"[fold {fold_id}] inference val_auc={val_auc:.3f}  (val_loss skipped -- no train_df given)")

    oof_df = pd.DataFrame(val_track)
    oof_df["fold"] = fold_id

    checkpoint_dir_for_fold = os.path.join(out_dir, f"fold-{fold_id}")
    os.makedirs(checkpoint_dir_for_fold, exist_ok=True)
    oof_df.to_csv(os.path.join(checkpoint_dir_for_fold, "oof_val_result_inference.csv"), index=False)
    return oof_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_pooled_oof(oof_dfs, out_dir, out_name="pooled_oof.csv"):
    pooled = pd.concat(oof_dfs, ignore_index=True)
    pooled.to_csv(os.path.join(out_dir, out_name), index=False)

    sample_auc = roc_auc_score(pooled["label"], pooled["probability"])
    subj_level = pooled.groupby("subject").agg(mean_prob=("probability", "mean"), label=("label", "first"))
    subject_auc = roc_auc_score(subj_level["label"], subj_level["mean_prob"])
    print(f"\nPooled OOF sample-level AUC: {sample_auc:.4f} (n={len(pooled)})")
    print(f"Pooled OOF subject-level AUC: {subject_auc:.4f} (n={len(subj_level)})")
    return sample_auc, subject_auc


def main(pretrained_encoder_dir, init_mode, num_frozen_blocks, linear_probe_only):
    config = VMAEConfig()
    out_dir = os.path.join(
        config.checkpoint_dir, init_mode,
        "linear_probe" if linear_probe_only else f"frozen_{num_frozen_blocks}blocks"
    )
    os.makedirs(out_dir, exist_ok=True)

    missing_folds = [
        f for f in range(1, 6)
        if not os.path.exists(os.path.join(pretrained_encoder_dir, init_mode, f"fold-{f}", "encoder"))
    ]
    if missing_folds:
        raise FileNotFoundError(
            f"Missing pretrained encoders ({init_mode}) for fold(s) {missing_folds}. Run "
            f"`python pretrain_videomae.py --fold <n>"
            f"{' --from_scratch' if init_mode == 'scratch' else ''}` for each first."
        )

    finetune_args = dict(pretrained_encoder_dir=pretrained_encoder_dir, init_mode=init_mode,
                          num_frozen_blocks=num_frozen_blocks, linear_probe_only=linear_probe_only)
    with open(os.path.join(out_dir, "finetune_args.yaml"), "w") as f:
        yaml.dump(finetune_args, f, default_flow_style=False, sort_keys=False)

    df = pd.read_csv(config.data_csv)
    oof_dfs = []
    for fold_id in range(1, 6):
        train_df = df[df["cv_fold"] != fold_id].reset_index(drop=True)
        val_df = df[df["cv_fold"] == fold_id].reset_index(drop=True)
        print(f"\n=== Fold {fold_id}/5 ({init_mode}) ===")
        oof_df = train_fold(fold_id, config, train_df, val_df, pretrained_encoder_dir, init_mode,
                             out_dir, num_frozen_blocks, linear_probe_only)
        oof_dfs.append(oof_df)

    evaluate_pooled_oof(oof_dfs, out_dir)


def run_inference(pretrained_encoder_dir, init_mode, num_frozen_blocks, linear_probe_only,
                   folds=None, checkpoint_name="best_combined_model",
                   pooled_out_name="pooled_oof_inference.csv"):
    """
    Inference-only counterpart to main(): instead of finetuning, load each
    fold's already-finetuned checkpoint and evaluate it once on that
    fold's validation split.

    `pretrained_encoder_dir`/`init_mode`/`num_frozen_blocks`/
    `linear_probe_only` are only used to reconstruct the same `out_dir`
    path main() would have used (so this looks in the right place for
    checkpoints) -- they do NOT affect model architecture here, since the
    checkpoint itself already has the finetuned weights and config baked
    in via `from_pretrained`.

    folds: None -> every fold (1..5, matching main()'s unconditional
    range(1, 6)); or an explicit list, e.g. [2, 4].

    Folds without a saved checkpoint are skipped with a warning rather
    than raising, so you can run this against whichever folds happen to
    have finished finetuning so far.
    """
    config = VMAEConfig()
    out_dir = os.path.join(
        config.checkpoint_dir, init_mode,
        "linear_probe" if linear_probe_only else f"frozen_{num_frozen_blocks}blocks"
    )

    folds_to_run = list(range(1, 6)) if folds is None else list(folds)
    invalid = [f for f in folds_to_run if f not in range(1, 6)]
    if invalid:
        raise ValueError(f"Invalid fold id(s) {invalid} -- must be in 1..5")
    print(f"Running inference on folds: {folds_to_run}  (out_dir={out_dir})")

    df = pd.read_csv(config.data_csv)
    oof_dfs = []
    for fold_id in folds_to_run:
        train_df = df[df["cv_fold"] != fold_id].reset_index(drop=True)
        val_df = df[df["cv_fold"] == fold_id].reset_index(drop=True)
        print(f"\n=== Fold {fold_id}/5 ({init_mode}, inference only) ===")
        try:
            oof_df = run_inference_fold(fold_id, config, val_df, out_dir,
                                         checkpoint_name=checkpoint_name, train_df=train_df)
        except FileNotFoundError as e:
            print(f"[fold {fold_id}] SKIPPED -- {e}")
            continue
        oof_dfs.append(oof_df)

    if not oof_dfs:
        print("No folds produced results (no checkpoints found) -- nothing to pool.")
        return None

    return evaluate_pooled_oof(oof_dfs, out_dir, out_name=pooled_out_name)


if __name__ == "__main__":
    config = VMAEConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_encoder_dir", default=None,
                         help="Defaults to VMAEConfig.checkpoint_dir")
    parser.add_argument("--from_scratch", action="store_true",
                         help="Finetune from the from-scratch-pretrained encoders instead of the "
                              "Kinetics-init ones -- must match how you ran pretrain_videomae.py")
    parser.add_argument("--num_frozen_blocks", type=int, default=None,
                         help="Defaults to VMAEConfig.num_frozen_blocks")
    parser.add_argument("--linear_probe_only", action="store_true")
    parser.add_argument("--infer", action="store_true",
                         help="Inference-only mode: load each fold's saved finetuned checkpoint "
                              "and evaluate it once on its validation split, instead of finetuning "
                              "from scratch.")
    parser.add_argument("--folds", nargs="+", default=None,
                         help="[--infer mode only] which fold(s) to run, e.g. '--folds 2 4'. "
                              "If omitted, runs all 5 folds.")
    parser.add_argument("--checkpoint_name", default="best_combined_model",
                         help="[--infer mode only] checkpoint subfolder name to load from each "
                              "fold's directory (matches what train_fold passed to "
                              "model.save_pretrained).")
    args = parser.parse_args()

    init_mode = "scratch" if args.from_scratch else "kinetics"
    num_frozen_blocks = args.num_frozen_blocks if args.num_frozen_blocks is not None else config.num_frozen_blocks
    linear_probe_only = args.linear_probe_only or config.linear_probe_only

    if args.infer:
        folds = [int(f) for f in args.folds] if args.folds is not None else None
        run_inference(
            args.pretrained_encoder_dir or config.checkpoint_dir,
            init_mode, num_frozen_blocks, linear_probe_only,
            folds=folds, checkpoint_name=args.checkpoint_name,
        )
    else:
        main(
            args.pretrained_encoder_dir or config.checkpoint_dir,
            init_mode, num_frozen_blocks, linear_probe_only,
        )
