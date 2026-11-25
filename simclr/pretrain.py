r"""
SimCLR-style self-supervised pretraining for the 3D ResNet50 video encoder,
adapted from https://github.com/Jefidev/Video-SimCLR/blob/master/main.py for
this project's data and backbone.

What changed from the reference repo, and why:
- Backbone: their I3D -> your resnet50(**resnet_param). The projection head
  and loss are architecture-agnostic, so this swap is straightforward.
- Augmentations: their color-jitter (brightness/contrast/saturation/hue) is
  built for natural RGB video. Ultrasound is grayscale speckle texture, so
  hue/saturation jitter doesn't map onto anything meaningful here. Swapped
  for: random crop+resize, mild rotation/shear/translation, Gaussian blur,
  and intensity jitter (brightness/contrast-like scaling) -- augmentations
  that correspond to real nuisance variation in ultrasound acquisition
  (probe angle, pressure, gain settings) rather than natural-image variation.
  No vertical flip -- as flagged earlier in this project, ultrasound has a
  meaningful up/down orientation (skin surface vs. depth) that flipping
  would destroy.
- Optimizer: their LARS -> AdamW. LARS is built for the very large batch
  sizes (256+) SimCLR papers use; with your dataset size, a large-batch
  optimizer doesn't buy you anything and adds a dependency.
- Loss: implemented NT-Xent directly (no pytorch_metric_learning dependency)
  so this script doesn't require anything beyond torch/opencv/pandas.
- No mlflow -- swapped for a plain CSV history log, consistent with the
  training_history.csv pattern used everywhere else in this project.
- Sampling: uses the SAME va.SlidingWindowSampler(clip_len, step) as
  train_cv.py's VideoDataset, rather than one random clip per video per
  epoch. This keeps the pretraining data distribution identical to what
  finetuning will see -- same clips, same per-video window counts, same
  relative weighting of long vs. short videos -- so pretraining isn't
  implicitly exposed to a different sampling of the data than finetuning is.

Important framing on WHICH videos to pretrain on:
This script supports two modes, via --fold:
- `--fold <n>` (recommended, and now the intended default workflow): restrict
  pretraining to fold n's TRAINING subjects only (cv_fold != n). This means
  fold n's SimCLR encoder never sees fold n's held-out subjects' videos at
  all -- not labels, not raw pixels, nothing. That makes each fold's
  finetuned model a genuinely clean held-out evaluation, with no caveat
  needed about the encoder having indirectly seen the test subjects. This
  requires running pretraining 5 times (once per fold) before finetuning,
  each producing its own checkpoints/simclr/fold-<n>/simclr_encoder.pt.
- omitting --fold: pretrains on ALL subjects at once, in one run. Cheaper
  (1 pretraining run instead of 5), but every fold's finetuned model then
  starts from an encoder that has seen that fold's held-out subjects'
  videos in an unsupervised, label-free way. Fine for quick experimentation,
  but if this ever goes into a writeup, don't describe the encoder as
  "never having seen the test subjects" under this mode -- it did, just
  without labels.

"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import vidaug.augmentors as va
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import ResnetConfig, SimCLRConfig
from datasets import img2tensor, imgResize, video_loader


# ---------------------------------------------------------------------------
# Two-view augmentation pipelines
# ---------------------------------------------------------------------------

def build_view_transform(config, view_seed_strength="a"):
    """
    Two independently-randomized augmentation pipelines. Both draw from the
    same pool of transforms so neither view is a strict subset/superset of
    the other -- that matters for SimCLR, since the model shouldn't be able
    to tell views apart by "which one has more augmentation."
    """
    return va.Sequential([
        va.Sometimes(0.8, va.RandomRotate(config.rotation_degree)),
        va.Sometimes(0.8, va.RandomTranslate(config.translation_degree, y=config.translation_degree)),
        va.Sometimes(0.5, va.RandomShear(config.shear_degree, config.shear_degree)),
        va.Sometimes(0.5, va.GaussianBlur(config.gaussian_blur_sigma)),
        va.Sometimes(0.6, va.HorizontalFlip()),
        # intensity jitter -- stands in for the acquisition-setting variation
        # (gain, probe pressure) that color jitter would represent in RGB video.
        # NOTE: verify these two augmentors exist under this name in your
        # installed vidaug version -- some vidaug forks expose intensity
        # augmentors under a separate submodule. If `va.Add`/`va.Multiply`
        # aren't found, the `--skip_intensity_jitter` flag below disables them.
        va.Sometimes(0.5, va.Add(int(255 * 0.08))),
        va.Sometimes(0.5, va.Multiply(1.15)),
    ])


class SimCLRVideoDataset(Dataset):
    """
    Unsupervised: no labels are read or used. Uses the SAME
    va.SlidingWindowSampler(clip_len, step) as train_cv.py's VideoDataset,
    so pretraining sees the identical set of (video, window) clips that
    finetuning will later train/validate on -- same distribution of clip
    positions per video, same relative contribution of long vs. short
    videos, no mismatch between the two stages.

    Each sampled window becomes ONE item, turned into TWO independently-
    augmented views (the SimCLR positive pair).
    """
    def __init__(self, video_paths, config, skip_intensity_jitter=False):
        self.config = config
        self.frame_cache = {}
        sampler = va.SlidingWindowSampler(clip_len=config.clip_len, step=config.step_size)

        # precompute every (video_path, frame_indices) pair up front, exactly
        # like VideoDataset._prepare() does in train_cv.py
        self.samples = []
        for path in video_paths:
            frames = self._get_frames(path)
            for window in sampler(len(frames)):
                self.samples.append((path, window))

        if skip_intensity_jitter:
            self.transform_a = va.Sequential([
                va.Sometimes(0.8, va.RandomRotate(config.rotation_degree)),
                va.Sometimes(0.8, va.RandomTranslate(config.translation_degree, y=config.translation_degree)),
                va.Sometimes(0.5, va.RandomShear(config.shear_degree, config.shear_degree)),
                va.Sometimes(0.5, va.GaussianBlur(config.gaussian_blur_sigma)),
                va.Sometimes(0.6, va.HorizontalFlip()),
            ])
            self.transform_b = va.Sequential([
                va.Sometimes(0.8, va.RandomRotate(config.rotation_degree)),
                va.Sometimes(0.8, va.RandomTranslate(config.translation_degree, y=config.translation_degree)),
                va.Sometimes(0.5, va.RandomShear(config.shear_degree, config.shear_degree)),
                va.Sometimes(0.5, va.GaussianBlur(config.gaussian_blur_sigma)),
                va.Sometimes(0.6, va.HorizontalFlip()),
            ])
        else:
            self.transform_a = build_view_transform(config, "a")
            self.transform_b = build_view_transform(config, "b")

    def _get_frames(self, path):
        if path not in self.frame_cache:
            self.frame_cache[path] = video_loader(path)
        return self.frame_cache[path]

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, clip, augmentor):
        aug_clip = augmentor(clip)
        resized = [imgResize(f, target_size=self.config.frame_size) for f in aug_clip]
        return img2tensor(np.array(resized), self.config.repeat_chn)

    def __getitem__(self, idx):
        path, frame_indices = self.samples[idx]
        frames = self._get_frames(path)
        clip = [frames[i] for i in frame_indices]
        xi = self._to_tensor(clip, self.transform_a)
        xj = self._to_tensor(clip, self.transform_b)
        return xi, xj


# ---------------------------------------------------------------------------
# Model: backbone + projection head
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """
    encoder: your resnet50(**resnet_param), with its final fc replaced by
             identity so it emits a feature vector instead of a class logit.
    projector: small MLP head, standard SimCLR design (2048 -> hidden -> proj_dim)
    """
    def __init__(self, encoder, feature_dim, proj_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)          # (B, feature_dim)
        z = self.projector(h)        # (B, proj_dim)
        return F.normalize(z, dim=1)


def build_encoder(config, from_scratch=False):
    """
    Builds the SAME resnet50 backbone used for classification, but with the
    final fc layer stripped off (Identity) so it emits a feature vector.

    from_scratch=False (default, recommended): loads Kinetics weights first,
        then SimCLR pretraining ADAPTS those features to ultrasound. This is
        domain adaptation of an already-competent feature extractor, not
        representation learning from zero -- much more realistic for a
        dataset this size (~38 subjects, ~800-1000 clips). SimCLR's original
        results relied on large batches and huge unlabeled corpora; without
        that scale, contrastive pretraining from random init risks collapsing
        to degenerate embeddings or learning shortcuts tied to the specific
        augmentation pipeline rather than real anatomical/motion structure.

    from_scratch=True: skips the Kinetics init entirely -- resnet50 starts
        from its default random initialization, and SimCLR has to learn
        useful features with no prior. Include this as an ABLATION to see
        how much of any SimCLR benefit is actually coming from Kinetics vs.
        the contrastive step itself -- don't expect this arm to win; if it
        does, that's a more surprising and worth-investigating result.
    """
    from models.resnet import resnet50

    resnet_param = dict(config.resnet_param)
    feature_dim = 2048  # standard ResNet50 penultimate feature size
    encoder = resnet50(**resnet_param)

    if from_scratch:
        print("Building encoder from SCRATCH (no Kinetics init) -- ablation mode")
    else:
        state_dict = torch.load(config.resnet_pretrained_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if not k.startswith("fc.")}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
                
        encoder.load_state_dict(state_dict, strict=False)
        print("Building encoder from Kinetics-pretrained weights")

    encoder.fc = nn.Identity()  # strip the classification head
    return encoder, feature_dim


# ---------------------------------------------------------------------------
# NT-Xent (SimCLR contrastive loss), implemented directly
# ---------------------------------------------------------------------------

def nt_xent_loss(zi, zj, temperature=0.1):
    """
    zi, zj: (B, D) L2-normalized embeddings of the two views.
    Standard SimCLR NT-Xent: for each of the 2B embeddings, its positive is
    the other view of the same clip; all other 2B-2 embeddings in the batch
    are negatives.
    """
    B = zi.shape[0]
    z = torch.cat([zi, zj], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # positive pairs: i <-> i+B
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    positives = sim[torch.arange(2 * B), pos_idx]

    denom = torch.logsumexp(sim, dim=1)
    loss = -(positives - denom).mean()
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def collect_video_paths(config, video_root=".", fold_id=None):
    """
    fold_id=None: every unique video in the dataset (original behavior).
    fold_id=<int>: only videos from subjects in that fold's TRAINING split
                   (cv_fold != fold_id) -- i.e. exactly the subjects that
                   fold's classifier will train on. This is what makes the
                   per-fold pretraining strict: fold i's encoder never sees
                   fold i's held-out subjects' videos at all, not even
                   label-free, so there's no sense in which the pretraining
                   stage has "seen" the subjects that fold is evaluated on.
    """
    df = pd.read_csv(config.data_csv)
    if fold_id is not None:
        df = df[df["cv_fold"] != fold_id]
    paths = df["Video"].drop_duplicates().apply(
        lambda p: os.path.join(video_root, p.replace("\\", os.sep))
    ).tolist()
    return paths


def pretrain(config, epochs=100, batch_size=16, lr=3e-4, weight_decay=1e-6,
             temperature=0.1, proj_dim=128, grad_clip_norm=5.0,
             video_root=".", checkpoint_dir="./checkpoints/simclr",
             skip_intensity_jitter=False, num_workers=2, fold_id=None,
             from_scratch=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # separate subdirectory per init mode -- keeps kinetics-init and
    # scratch-init runs from overwriting each other's checkpoints, since
    # you'll want both around for the comparison
    checkpoint_dir = os.path.join(checkpoint_dir, "scratch" if from_scratch else "kinetics")
    if fold_id is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f"fold-{fold_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save every argument this run was called with -- makes it possible to
    # look back at any checkpoint later and know exactly how it was produced,
    # without having to remember or reconstruct the command line
    run_args = dict(
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        temperature=temperature, proj_dim=proj_dim, grad_clip_norm=grad_clip_norm,
        video_root=video_root, checkpoint_dir=checkpoint_dir,
        skip_intensity_jitter=skip_intensity_jitter, num_workers=num_workers,
        fold_id=fold_id, from_scratch=from_scratch,
        clip_len=config.clip_len, step_size=config.step_size, frame_size=config.frame_size,
    )
    with open(os.path.join(checkpoint_dir, "pretrain_args.yaml"), "w") as f:
        yaml.dump(run_args, f, default_flow_style=False, sort_keys=False)

    video_paths = collect_video_paths(config, video_root, fold_id=fold_id)
    scope = f"fold {fold_id}'s training subjects only" if fold_id is not None else "ALL subjects (no fold restriction)"
    init_desc = "from scratch" if from_scratch else "from Kinetics"
    print(f"Pretraining ({init_desc}) on {len(video_paths)} unique videos ({scope}, unsupervised -- no labels used)")

    dataset = SimCLRVideoDataset(video_paths, config, skip_intensity_jitter=skip_intensity_jitter)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=num_workers)

    encoder, feature_dim = build_encoder(config, from_scratch=from_scratch)
    model = SimCLRModel(encoder, feature_dim, proj_dim=proj_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xi, xj in tqdm(dataloader, desc=f"SimCLR epoch {epoch}"):
            xi = xi.to(device)
            xj = xj.to(device)

            optimizer.zero_grad()
            zi = model(xi)
            zj = model(xj)
            loss = nt_xent_loss(zi, zj, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}: avg NT-Xent loss = {avg_loss:.4f}  lr = {curr_lr:.2e}")
        history.append({"epoch": epoch, "loss": avg_loss, "lr": curr_lr})

        # save every epoch -- pretraining runs can be long; don't lose progress
        torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, "simclr_encoder.pt"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "simclr_full.pt"))
        pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, "simclr_pretrain_history.csv"), index=False)

    print(f"\nDone. Encoder weights saved to {os.path.join(checkpoint_dir, 'simclr_encoder.pt')}")
    print("Run this once per fold (--fold 1 through 5) before finetuning, so each "
          "fold's classifier starts from an encoder that never saw that fold's "
          "held-out subjects -- pass --simclr_checkpoint_dir to finetune_simclr.py "
          "(it will look under fold-<n>/ automatically).")


if __name__ == "__main__":
    simclr_config = SimCLRConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None,
                         help="Restrict pretraining to this fold's TRAINING subjects only "
                              "(cv_fold != fold). Omit to pretrain on all subjects instead.")
    parser.add_argument("--from_scratch", action="store_true",
                         help="Skip Kinetics init; encoder starts randomly initialized. "
                              "Ablation mode -- expect this to underperform the default "
                              "Kinetics-init unless you have a lot more unlabeled data.")
    # All of these default to None so that, unless explicitly passed on the
    # command line, the value comes from SimCLRConfig instead -- the config
    # file is the source of truth; CLI flags are overrides for one-off runs.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--video_root", default=".")
    parser.add_argument("--skip_intensity_jitter", action="store_true",
                         help="Use only geometric augmentations if va.Add/va.Multiply "
                              "aren't available in your vidaug install")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    def resolve(cli_value, config_attr):
        """CLI flag wins if the user passed one; otherwise fall back to SimCLRConfig."""
        return cli_value if cli_value is not None else getattr(simclr_config, config_attr)

    pretrain(
        simclr_config,
        epochs=resolve(args.epochs, "epochs"),
        batch_size=resolve(args.batch_size, "batch_size"),
        lr=resolve(args.lr, "lr"),
        weight_decay=resolve(args.weight_decay, "weight_decay"),
        temperature=resolve(args.temperature, "temperature"),
        proj_dim=resolve(args.proj_dim, "proj_dim"),
        grad_clip_norm=resolve(args.grad_clip_norm, "grad_clip_norm"),
        checkpoint_dir=resolve(args.checkpoint_dir, "checkpoint_dir"),
        video_root=args.video_root,
        skip_intensity_jitter=True,
        num_workers=args.num_workers,
        fold_id=args.fold,
        from_scratch=args.from_scratch,
    )