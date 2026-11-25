r"""
VideoMAE (ViT-based masked autoencoder) pretraining, fold-restricted, following
the method of https://github.com/MCG-NJU/VideoMAE, applied to ultrasound data.

Uses HuggingFace `transformers`' VideoMAEForPreTraining -- the exact same
architecture and weights as the official MCG-NJU release, loaded via
`from_pretrained("MCG-NJU/videomae-base")` -- rather than a hand-rolled
reimplementation of the tubelet embedding / transformer / decoder. This
matters: VideoMAEForPreTraining internally unnormalizes pixel values with
ImageNet mean/std and computes the masked-patch reconstruction MSE loss
itself, which is easy to get subtly wrong by hand.

Requires: pip install transformers
Requires internet access (once) to download MCG-NJU/videomae-base from the
HuggingFace Hub -- unlike the locally-vendored Kinetics-ResNet50 checkpoint,
this is not something you provide a local path for; huggingface_hub caches
it after the first download.

Key differences from the ResNet50/SimCLR pipeline, worth being aware of:
- Input tensor shape is (batch, num_frames, channels, height, width) --
  frames and channels are in the OPPOSITE order from the ResNet pipeline's
  (batch, channels, frames, height, width).
- num_frames=16, patch_size=16, tubelet_size=2, image_size=224 come from the
  checkpoint's own config (read off model.config at runtime), not from
  Config.clip_len -- this pipeline's clip length is decoupled from both the
  ResNet arm and the SimCLR arm, same reasoning as those two are decoupled
  from each other.
- Pretraining objective is masked-patch RECONSTRUCTION (MSE on pixel
  values), not contrastive (NT-Xent) -- so there's no "two views per clip"
  dataset here, just one clip plus a random tube mask built fresh each step.

Fold-restriction: for fold k, pretraining uses ONLY videos from fold k's
TRAINING subjects (cv_fold != k) -- same discipline as pretrain_simclr.py,
so fold k's encoder never sees fold k's held-out subjects at any stage.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import vidaug.augmentors as va
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import VMAEConfig
from datasets import imgResize, video_loader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collect_video_paths(config, video_root=".", fold_id=None):
    """Same fold-restriction pattern as pretrain_simclr.py's collect_video_paths."""
    df = pd.read_csv(config.data_csv)
    if fold_id is not None:
        df = df[df["cv_fold"] != fold_id]
    return df["Video"].drop_duplicates().apply(
        lambda p: os.path.join(video_root, p.replace("\\", os.sep))
    ).tolist()


def frame_to_normalized_tensor(frames, image_size):
    """
    frames: list of grayscale (H, W) uint8 arrays (a sampled clip)
    Returns: (T, C, H, W) float tensor, resized, replicated to 3 channels,
    and normalized with ImageNet mean/std -- matching what
    VideoMAEForPreTraining expects to internally unnormalize.
    """
    resized = [imgResize(f, target_size=image_size) for f in frames]
    arr = np.stack(resized).astype(np.float32) / 255.0  # (T, H, W)
    if arr.ndim == 3:
        arr = arr[..., None]                             # (T, H, W, 1)
    arr = np.repeat(arr, 3, axis=-1)                       # (T, H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (T, C, H, W)
    return tensor


class VideoMAEPretrainDataset(Dataset):
    """
    Unsupervised, single-view (unlike SimCLR's two views) -- MAE's
    "augmentation" is the mask itself, built fresh per training step, not
    per dataset item. Uses the SAME sliding-window sampling philosophy as
    pretrain_simclr.py (precompute every (video, window) pair up front) so
    per-video window counts stay consistent with the rest of this project's
    pipelines.
    """
    def __init__(self, video_paths, num_frames, step_size, image_size, augment=False, config=None):
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_cache = {}
        sampler = va.SlidingWindowSampler(clip_len=num_frames, step=step_size)

        self.samples = []
        for path in video_paths:
            frames = self._get_frames(path)
            for window in sampler(len(frames)):
                self.samples.append((path, window))

        # VideoMAE's original recipe relies on masking itself for
        # regularization and uses only light augmentation (random crop,
        # flip) -- kept optional and off by default to match that, rather
        # than the heavier augmentation pipeline used for the classifier.
        self.augment = augment
        if augment:
            self.transform = va.Sequential([
                va.Sometimes(0.5, va.HorizontalFlip()),
                va.Sometimes(0.5, va.RandomTranslate(config.translation_degree if config else 10,
                                                      y=config.translation_degree if config else 10)),
            ])

    def _get_frames(self, path):
        if path not in self.frame_cache:
            self.frame_cache[path] = video_loader(path)
        return self.frame_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame_indices = self.samples[idx]
        frames = self._get_frames(path)
        clip = [frames[i] for i in frame_indices]
        if self.augment:
            clip = self.transform(clip)
        return frame_to_normalized_tensor(clip, self.image_size)


# ---------------------------------------------------------------------------
# Tube masking
# ---------------------------------------------------------------------------

def build_tube_mask_batch(batch_size, num_patches_per_frame, num_temporal_groups, mask_ratio, device):
    """
    "Tube masking": the SAME spatial mask is applied to every temporal group
    (this is what the VideoMAE paper found works best -- masking randomly
    per-frame lets the model cheat by copying an unmasked neighboring
    frame's patch, since consecutive ultrasound/video frames are highly
    redundant; tube masking removes that shortcut).

    Returns: (batch_size, num_patches_per_frame * num_temporal_groups) bool
    tensor, matching what VideoMAEForPreTraining.forward(bool_masked_pos=...)
    expects.
    """
    n_mask = int(round(num_patches_per_frame * mask_ratio))
    masks = []
    for _ in range(batch_size):
        spatial_mask = torch.zeros(num_patches_per_frame, dtype=torch.bool)
        idx = torch.randperm(num_patches_per_frame)[:n_mask]
        spatial_mask[idx] = True
        full_mask = spatial_mask.unsqueeze(0).repeat(num_temporal_groups, 1).reshape(-1)
        masks.append(full_mask)
    return torch.stack(masks).to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(fold_id, config, video_root=".", epochs=None, batch_size=None, lr=None,
             weight_decay=None, mask_ratio=None, warmup_epochs=None, grad_clip_norm=None,
             num_workers=2, from_scratch=False):
    from transformers import VideoMAEConfig as HFVideoMAEConfig, VideoMAEForPreTraining

    def resolve(v, attr):
        return v if v is not None else getattr(config, attr)

    epochs = resolve(epochs, "epochs")
    batch_size = resolve(batch_size, "batch_size")
    lr = resolve(lr, "lr")
    weight_decay = resolve(weight_decay, "weight_decay")
    mask_ratio = resolve(mask_ratio, "mask_ratio")
    warmup_epochs = resolve(warmup_epochs, "warmup_epochs")
    grad_clip_norm = resolve(grad_clip_norm, "grad_clip_norm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # separate subdirectory per init mode -- same reasoning as SimCLR's
    # kinetics/ vs scratch/ split: keeps the two lineages from overwriting
    # each other's checkpoints, since you'll want both around to compare
    checkpoint_dir = os.path.join(config.checkpoint_dir, "scratch" if from_scratch else "kinetics",
                                   f"fold-{fold_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_args = dict(fold_id=fold_id, from_scratch=from_scratch, epochs=epochs, batch_size=batch_size,
                     lr=lr, weight_decay=weight_decay, mask_ratio=mask_ratio,
                     warmup_epochs=warmup_epochs, grad_clip_norm=grad_clip_norm,
                     pretrained_model_name=config.pretrained_model_name,
                     step_size=config.step_size, checkpoint_dir=checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "pretrain_args.yaml"), "w") as f:
        yaml.dump(run_args, f, default_flow_style=False, sort_keys=False)

    if from_scratch:
        # Fetch only the ARCHITECTURE config (a small JSON: hidden sizes,
        # depth, patch/tubelet settings) -- not the pretrained weight file --
        # then build a randomly-initialized model from it. This keeps the
        # from-scratch arm architecturally identical to the Kinetics-init
        # arm (fair comparison), rather than guessing hyperparameters by hand.
        print(f"Fetching architecture config from {config.pretrained_model_name} "
              f"(config only, NOT the pretrained weights)...")
        hf_config = HFVideoMAEConfig.from_pretrained(config.pretrained_model_name)
        model = VideoMAEForPreTraining(hf_config).to(device)
        print("Built VideoMAEForPreTraining from a RANDOM initialization -- ablation mode. "
              "ViT-based masked autoencoding is generally MORE data-hungry than CNN-based "
              "methods; with ~800-1000 clips this arm is a real long shot, not an "
              "expected winner -- see the module docstring.")
    else:
        print(f"Loading {config.pretrained_model_name} (downloads on first use)...")
        model = VideoMAEForPreTraining.from_pretrained(config.pretrained_model_name).to(device)

    num_frames = model.config.num_frames
    image_size = model.config.image_size
    patch_size = model.config.patch_size
    tubelet_size = model.config.tubelet_size
    num_patches_per_frame = (image_size // patch_size) ** 2
    num_temporal_groups = num_frames // tubelet_size
    print(f"Checkpoint config: num_frames={num_frames} image_size={image_size} "
          f"patch_size={patch_size} tubelet_size={tubelet_size} "
          f"(seq_length={num_patches_per_frame * num_temporal_groups})")

    video_paths = collect_video_paths(config, video_root, fold_id=fold_id)
    init_desc = "from scratch" if from_scratch else "from Kinetics"
    print(f"Pretraining ({init_desc}) on {len(video_paths)} unique videos (fold {fold_id}'s training "
          f"subjects only, unsupervised -- no labels used)")

    dataset = VideoMAEPretrainDataset(video_paths, num_frames, config.step_size, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                       milestones=[warmup_epochs])

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for pixel_values in tqdm(dataloader, desc=f"VideoMAE fold {fold_id} epoch {epoch}"):
            pixel_values = pixel_values.to(device)
            B = pixel_values.shape[0]
            bool_masked_pos = build_tube_mask_batch(B, num_patches_per_frame, num_temporal_groups,
                                                     mask_ratio, device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}: avg reconstruction MSE = {avg_loss:.4f}  lr = {curr_lr:.2e}")
        history.append({"epoch": epoch, "loss": avg_loss, "lr": curr_lr})

        # save every epoch -- pretraining runs can be long; don't lose progress.
        # Save the encoder only (model.videomae) for finetuning -- discard the
        # decoder, it's only needed during pretraining reconstruction.
        model.videomae.save_pretrained(os.path.join(checkpoint_dir, "encoder"))
        pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, "pretrain_history.csv"), index=False)

    print(f"\nDone. Encoder saved to {os.path.join(checkpoint_dir, 'encoder')}")
    print(f"Run this once per fold (--fold 1 through 5) before finetuning "
          f"(pass the SAME --from_scratch flag to finetune_videomae.py so it looks in the right subdirectory).")


if __name__ == "__main__":
    config = VMAEConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True,
                         help="Restrict pretraining to this fold's TRAINING subjects only (cv_fold != fold)")
    parser.add_argument("--from_scratch", action="store_true",
                         help="Skip the Kinetics checkpoint entirely -- build the model with the "
                              "same architecture but random weights. Ablation mode: ViT-based "
                              "masked autoencoding is generally MORE data-hungry than CNN-based "
                              "SimCLR, so expect this to underperform the Kinetics-init default "
                              "given only ~800-1000 clips. Still fetches the checkpoint's config "
                              "(architecture only, a small JSON) for a fair apples-to-apples "
                              "architecture comparison -- does not download the pretrained weights.")
    parser.add_argument("--video_root", default=".")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    pretrain(
        args.fold, config, video_root=args.video_root, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio, warmup_epochs=args.warmup_epochs,
        grad_clip_norm=args.grad_clip_norm, num_workers=args.num_workers,
        from_scratch=args.from_scratch,
    )