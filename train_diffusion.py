"""
Diffusion-prior pretraining, cleaned up and switched to the new data split.

What changed from the old train_diffusion.py / config_old.py:

- Removed `main()`. It referenced `config.train_fold`,
  `config.specify_fold_test_patient`, and a numeric `Patient {id}` naming
  scheme that don't exist anywhere in Diffusion2DConfig -- it could not
  have run as written.
- Removed `VideoFrameDataset` (pre-extracted frame images on disk) and
  `train_all()`, which used a hardcoded numeric patient-ID list
  (`train_id_latent`) and a `cropped_frames/<subject>/` directory
  layout. Neither corresponds to the actual `Study ID` / `Video` columns
  used everywhere else in this project. Kept only `Video2FrameDataset`,
  which extracts frames directly from the video paths in the CSV via
  `video_loader` -- the same data path the classifier pipeline uses, so
  both pipelines read the exact same source of truth.
- Removed `process_cv_csv()`'s dependency on 5 separate per-fold CSVs.
  Folds now come from the single combined CSV's `cv_fold` column (see
  build_folds.py), which is the same subject-level, source+label-stratified
  split used for the classifier -- so the diffusion prior for fold i is
  trained on exactly the subjects the classifier's fold i also trains on,
  and never sees fold i's validation subjects.
- `trainer.load(...)` is now only called when a milestone is actually set,
  instead of unconditionally (which would fail with `milestone=None`).
"""
import os
from pathlib import Path
import numpy as np

import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

from config import DiffusionConfig
from datasets import video_loader
from cohort import healthy_latent_cohort
from reproducibility import seed_training, fold_seed
from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    GaussianDiffusion,
    Trainer,
    Unet,
)


class Video2FrameDataset(torch.utils.data.Dataset):
    """Retain evenly spaced grayscale frames, capped per video."""

    def __init__(self, video_paths, transforms=None, max_frames_per_video=32):
        if max_frames_per_video < 1:
            raise ValueError('max_frames_per_video must be positive')
        self.transforms = transforms
        self.frames = []
        for path in video_paths:
            frames = video_loader(str(path))
            if not frames:
                raise ValueError(f'Empty or unreadable video: {path}')
            indices = np.linspace(0, len(frames) - 1, min(len(frames), max_frames_per_video), dtype=int)
            self.frames.extend(frames[i] for i in indices)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        image = self.frames[idx]
        if self.transforms:
            image = self.transforms(image)
        return image


def get_model(config: DiffusionConfig):
    unet = Unet(**config.unet_param)
    return GaussianDiffusion(model=unet, image_size=config.frame_size, **config.diffusion_param)


def get_transform(config: DiffusionConfig):
    a = config.aug_param
    return T.Compose([
        T.Resize((config.frame_size, config.frame_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(a["random_horizontal_flip_prob"]),
        T.RandomVerticalFlip(a["random_vertical_flip_prob"]),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=a["random_gaussian_blur_kernel"], sigma=a["random_gaussian_blur_sigma"])],
            p=a["random_gaussian_blur_prob"],
        ),
        T.RandomAffine(
            degrees=a["random_affine_degree"],
            translate=a["random_affine_translate"],
            scale=a["random_affine_scale"],
            shear=a["random_affine_shear"],
        ),
        T.ToTensor(),
    ])


def config_to_dict(config):
    cls = config if isinstance(config, type) else type(config)
    values = {}
    for parent in reversed(cls.__mro__):
        values.update(vars(parent))
    if not isinstance(config, type):
        values.update(vars(config))
    return {k: v for k, v in values.items() if not k.startswith('__') and not callable(v)}


def train_fold(config: DiffusionConfig, train_video_paths, fold_id):
    effective_seed = seed_training(fold_seed(config.seed, fold_id))
    print(f'[fold {fold_id}] diffusion training seed={effective_seed}')
    fold_ckpt = os.path.join(config.checkpoint_dir, f"fold-{fold_id}")
    os.makedirs(fold_ckpt, exist_ok=True)

    model = get_model(config)
    transform = get_transform(config)
    dataset = Video2FrameDataset(train_video_paths, transform, config.max_frames_per_video)
    print(f"[fold {fold_id}] training frames: {len(dataset)} (from {len(train_video_paths)} videos)")

    trainer = Trainer(
        diffusion_model=model,
        dataset=dataset,
        train_batch_size=config.batch_size,
        results_folder=fold_ckpt,
        data_seed=effective_seed,
        **config.trainer_param,
    )
    if config.milestone is not None:
        trainer.load(config.milestone)

    trainer.train()


def train_cv(config: DiffusionConfig):
    seed_training(config.seed)
    df = healthy_latent_cohort(pd.read_csv(config.data_csv), config.n_fold)
    root = Path(config.checkpoint_dir)
    if root.exists() and any(root.glob('fold-*/model-*.pt')):
        raise FileExistsError('Use a new checkpoint_dir for this pretraining protocol; existing weights are protected')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)
    df.to_csv(root / 'training_cohort.csv', index=False)

    for fold_id in range(1, config.n_fold + 1):
        # diffusion prior for fold i trains on the SAME subjects the
        # classifier's fold i trains on -- excludes fold i's val subjects
        train_video_paths = df.loc[df["cv_fold"] != fold_id, "Video"].tolist()
        fold_dir = root / f'fold-{fold_id}'
        fold_dir.mkdir(exist_ok=True)
        df.loc[df.cv_fold != fold_id].to_csv(fold_dir / 'training_manifest.csv', index=False)
        train_fold(config, train_video_paths, fold_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    config = DiffusionConfig()
    if args.seed is not None:
        config.seed = args.seed
    train_cv(config)
