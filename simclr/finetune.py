r"""
Finetune the classifier starting from SimCLR-pretrained weights.

Unlike an earlier version of this script, this does NOT monkey-patch
train_cv.py's get_model/train_fold/main. It has its OWN train_fold_simclr
and main, tailored specifically to a SimCLR-initialized backbone. That's
more code duplicated here than the monkey-patch version had, but it's
explicit: everything that happens to the model, and in what order, is
fully visible in this one file, without needing to trace through
train_cv.py's internals to find where the patched functions get called,
or worry about a patch silently breaking if train_cv.py's internals
change shape later.

Reused AS-IS from train_cv.py (generic dataloading/optimization utilities
with nothing SimCLR-specific about them):
    get_dataloader, get_param_groups, get_lr_scheduler, EMATracker,
    evaluate_pooled_oof, config_to_dict, DataWrapper

NOT reused -- reimplemented here, tailored to SimCLR:
    get_model   -> get_model_from_simclr (loads SimCLR weights, not Kinetics)
    train_fold  -> train_fold_simclr (calls get_model_from_simclr instead)
    main        -> main (always uses SimCLRFinetuneConfig; no config_class
                   parameter needed since this script only ever does the
                   one thing)

Config: runs under SimCLRFinetuneConfig (config.py), which inherits
ResnetConfig's hyperparameters but has its OWN checkpoint_dir
(./checkpoints/simclr_finetune_cv_12f) -- separate from both the plain
Kinetics-init ResnetConfig baseline's checkpoint_dir and from
SimCLRConfig's own PREtraining checkpoint_dir (this script loads FROM the
latter, but never writes into it).

Fold-awareness: each fold loads its OWN SimCLR checkpoint, from
<simclr_checkpoint_dir>/fold-<n>/simclr_encoder.pt. This matters -- if fold 3's
SimCLR encoder were pretrained on all subjects (or on some other fold's split),
fold 3's finetuned model would start from weights that have indirectly "seen"
fold 3's held-out subjects. Pretrain each fold's encoder separately first
(pretrain_simclr.py --fold 1, --fold 2, ... --fold 5) before running this.

Usage:
    python -m simclr.finetune --simclr_checkpoint_dir checkpoints/simclr
"""
import argparse
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from tqdm import tqdm

from train_cv import (
    get_dataloader, get_param_groups, get_lr_scheduler, EMATracker,
    evaluate_pooled_oof, config_to_dict, DataWrapper,
)
from config import SimCLRConfig, SimCLRFinetuneConfig


# ---------------------------------------------------------------------------
# Model: SimCLR-initialized backbone
# ---------------------------------------------------------------------------

def get_model_from_simclr(config, device, fold_id, simclr_checkpoint_dir, init_mode):
    """
    Same shape as train_cv.get_model(), except the backbone starts from
    THIS fold's SimCLR weights instead of the Kinetics checkpoint.

    init_mode: "kinetics" or "scratch" -- must match whichever mode you
    pretrained with (pretrain_simclr.py saves each mode under its own
    subdirectory precisely so these can't accidentally get mixed up).

    Freezing behavior is UNCHANGED from train_cv.get_model(): whatever
    `config.freeze_layers` says (conv1/layer1/layer2/layer3 frozen, layer4
    trainable, per the earlier fix) still applies here. SimCLR only changes
    what the WEIGHTS are initialized to, not which of them get gradients.
    """
    from models.resnet import resnet50

    resnet_model = resnet50(**config.resnet_param)

    simclr_checkpoint_path = os.path.join(simclr_checkpoint_dir, init_mode, f"fold-{fold_id}", "simclr_encoder.pt")
    print("Loading checkpoint from ", simclr_checkpoint_path)
    if not os.path.exists(simclr_checkpoint_path):
        raise FileNotFoundError(
            f"No SimCLR checkpoint at {simclr_checkpoint_path} -- run "
            f"`python pretrain_simclr.py --fold {fold_id}"
            f"{' --from_scratch' if init_mode == 'scratch' else ''}` first."
        )
    simclr_state = torch.load(simclr_checkpoint_path, map_location=device)
    # strict=False because the SimCLR encoder has no fc layer (it was
    # replaced with nn.Identity() during pretraining) -- that's expected;
    # the classification fc layer stays randomly initialized and gets
    # trained fresh during finetuning regardless.
    missing, unexpected = resnet_model.load_state_dict(simclr_state, strict=False)
    print(f"[fold {fold_id}, {init_mode}] loaded SimCLR weights from {simclr_checkpoint_path}")
    print(f"  missing keys (expected: just 'fc.*'): {missing}")
    print(f"  unexpected keys (should be empty): {unexpected}")

    for name, param in resnet_model.named_parameters():
        if name.startswith(config.freeze_layers):
            param.requires_grad = False

    model = DataWrapper(resnet_model)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Train / validate one fold -- tailored copy of train_cv.py's train_fold
# ---------------------------------------------------------------------------

def train_fold_simclr(config, train_df, val_df, fold_id, simclr_checkpoint_dir, init_mode):
    """
    Tailored copy of train_cv.py's train_fold, with exactly one
    substantive change: the model comes from get_model_from_simclr (this
    fold's SimCLR checkpoint) instead of train_cv.get_model (Kinetics
    init). Everything else -- dataloaders, pos_weight, optimizer/scheduler,
    grad clipping, EMA-smoothed checkpoint selection, saved outputs -- is
    identical to train_cv.py's own train_fold, duplicated here so this
    file has no dependency on monkey-patching train_cv's internals.
    """
    device = torch.device("cuda")
    model = get_model_from_simclr(config, device, fold_id, simclr_checkpoint_dir, init_mode)

    train_dl = get_dataloader(config, train_df, is_train=True)
    val_dl = get_dataloader(config, val_df, is_train=False)
    print(f"[SimCLR fold {fold_id}] train clips: {len(train_dl.dataset)}  val clips: {len(val_dl.dataset)}")

    # pos_weight from this fold's actual train-set balance (folds differ in
    # class ratio, so a single global constant would be wrong for some folds)
    n_pos = sum(train_df["label"])
    n_neg = len(train_df) - n_pos
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
    best_val_probs, best_val_labels, best_val_subjects, best_val_video_id = None, None, None, None

    for epoch in range(config.epochs):
        model.train()
        train_loss, train_steps = 0.0, 0
        grad_norms = []
        for data in tqdm(train_dl, desc=f"[SimCLR] fold {fold_id} epoch {epoch} train"):
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

        train_loss /= train_steps
        scheduler.step()

        model.eval()
        val_loss, val_steps = 0.0, 0
        val_probs, val_labels, val_subjects, val_video_id = [], [], [], []
        with torch.no_grad():
            for data in tqdm(val_dl, desc=f"[SimCLR] fold {fold_id} epoch {epoch} val"):
                labels = data["label"].to(device).float()
                B = labels.shape[0]
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * B
                val_steps += B

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs.reshape(-1).tolist())
                val_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
                val_subjects.extend(data["subject"])
                val_video_id.extend(data["video_id"])

        val_loss /= val_steps
        val_auc = roc_auc_score(val_labels, val_probs)
        ema_val_loss, ema_val_auc, combined_score = tracker.update(val_loss, val_auc)
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"[SimCLR fold {fold_id}] epoch {epoch}: train_loss={train_loss:.3f} "
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

        if tracker.is_best(combined_score):
            torch.save({"state_dict": model.state_dict()},
                       os.path.join(checkpoint_path, "best_combined_model.pth"))
            # this epoch's val predictions become this fold's OOF predictions
            best_val_probs, best_val_labels, best_val_subjects = val_probs, val_labels, val_subjects
            best_val_video_id = val_video_id
            print(f"[SimCLR fold {fold_id}] new best combined_score={combined_score:.4f} at epoch {epoch}")

    pd.DataFrame(history).to_csv(os.path.join(checkpoint_path, "training_history.csv"), index=False)

    oof_df = pd.DataFrame({
        "subject": best_val_subjects,
        "label": best_val_labels,
        "probability": best_val_probs,
        "video_id": best_val_video_id,
        "fold": fold_id,
    })
    oof_df.to_csv(os.path.join(checkpoint_path, "oof_val_result.csv"), index=False)
    return oof_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(simclr_checkpoint_dir, init_mode):
    # Verify all 5 fold checkpoints exist BEFORE starting any training --
    # fail fast rather than discovering fold 4 is missing after fold 1-3
    # already trained.
    missing_folds = [
        f for f in range(1, 6)
        if not os.path.exists(os.path.join(simclr_checkpoint_dir, init_mode, f"fold-{f}", "simclr_encoder.pt"))
    ]
    if missing_folds:
        raise FileNotFoundError(
            f"Missing SimCLR checkpoints ({init_mode}) for fold(s) {missing_folds}. Run "
            f"`python pretrain_simclr.py --fold <n>"
            f"{' --from_scratch' if init_mode == 'scratch' else ''}` for each missing fold first."
        )

    config = SimCLRFinetuneConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # config.yaml describes the classification-finetuning hyperparameters
    # (lr, epochs, freeze_layers, ...); finetune_simclr_args.yaml describes
    # WHICH SimCLR checkpoints were used to initialize each fold. Together,
    # in the same directory, they fully describe how this run was produced.
    with open(os.path.join(config.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)

    finetune_args = dict(
        simclr_checkpoint_dir=simclr_checkpoint_dir,
        init_mode=init_mode,
        simclr_checkpoint_paths={
            f: os.path.join(simclr_checkpoint_dir, init_mode, f"fold-{f}", "simclr_encoder.pt")
            for f in range(1, 6)
        },
    )
    with open(os.path.join(config.checkpoint_dir, "finetune_simclr_args.yaml"), "w") as f:
        yaml.dump(finetune_args, f, default_flow_style=False, sort_keys=False)

    df = pd.read_csv(config.data_csv)
    df["sex_label"] = np.where(df["Sex"] == "M", 0, 1)

    oof_dfs = []
    for fold_id in range(1, config.n_fold + 1):
        train_df = df[df["cv_fold"] != fold_id].reset_index(drop=True)
        val_df = df[df["cv_fold"] == fold_id].reset_index(drop=True)

        print(f"\n=== Fold {fold_id}/{config.n_fold} (SimCLR, {init_mode}) ===")
        print(f"train subjects={train_df['Study ID'].nunique()} (n_pos={train_df['label'].sum()}) "
              f"val subjects={val_df['Study ID'].nunique()} (n_pos={val_df['label'].sum()})")

        t0 = time.time()
        oof_df = train_fold_simclr(config, train_df, val_df, fold_id, simclr_checkpoint_dir, init_mode)
        oof_dfs.append(oof_df)
        print(f"[fold {fold_id}] done in {timedelta(seconds=int(time.time() - t0))}")

    evaluate_pooled_oof(oof_dfs, config.checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simclr_checkpoint_dir", default=SimCLRConfig().checkpoint_dir,
                         help="Base dir containing kinetics/ and/or scratch/ subdirectories, "
                              "each with fold-1/ through fold-5/ (from pretrain_simclr.py --fold <n>). "
                              "Defaults to SimCLRConfig.checkpoint_dir -- the PREtraining "
                              "checkpoint dir this script loads FROM, not to be confused with "
                              "SimCLRFinetuneConfig.checkpoint_dir, which is where THIS script's "
                              "own classification checkpoints get written TO.")
    parser.add_argument("--init_mode", choices=["kinetics", "scratch"], default="kinetics",
                         help="Which pretraining lineage to finetune from -- must match "
                              "how you ran pretrain_simclr.py (default, or --from_scratch)")
    args = parser.parse_args()
    main(args.simclr_checkpoint_dir, args.init_mode)