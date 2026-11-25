r"""
extract_fold_embeddings.py
===========================
Companion to simclr_pretrain.py. Loads a SPECIFIC fold's trained SimCLR
checkpoint (checkpoints/simclr/<kinetics|scratch>/fold-<n>/simclr_full.pt)
and extracts embeddings for that fold's clips, split into:

    - "train_subjects"     : cv_fold != n   -> subjects the encoder DID see
                              during pretraining (unsupervised, no labels)
    - "held_out_subjects"  : cv_fold == n   -> subjects the encoder NEVER
                              saw at all, in any form

This split is the whole point of your per-fold pretraining design (see the
docstring in simclr_pretrain.py): the meaningful test of representation
quality is how the embeddings look for held_out_subjects, since that's the
condition finetuning will actually face for this fold.

Reuses your own code directly (build_encoder, SimCLRModel, SimCLRVideoDataset,
build_view_transform) rather than reimplementing it, so the extracted
embeddings are produced by the exact same preprocessing/augmentation path as
training -- run this from the same project directory as simclr_pretrain.py.

Two kinds of embeddings are saved:
    - encoder features h (2048-d, post-encoder / pre-projector): what
      finetuning actually starts from. Used for kNN / linear probe /
      clustering / t-SNE quality checks.
    - projector output z (proj_dim-d, L2-normalized, two augmented views per
      clip): the space NT-Xent actually optimizes. Used for alignment /
      uniformity (Wang & Isola 2020). Only computed if --extract_pairs is set.

Output (written to --outdir):
    embeddings.npy       (N, 2048)  encoder features h, one row per clip
    labels.npy           (N,)
    groups.npy           (N,)       subject/source-prefix IDs
    splits.npy           (N,)       "train_subjects" / "held_out_subjects"
    video_paths.npy      (N,)       for traceability back to raw data
    z_view1.npy, z_view2.npy   (N, proj_dim)   only if --extract_pairs

Usage
-----
    python -m simclr.extract_embeddings --fold 2 \
        --checkpoint_dir ./checkpoints/simclr --init_mode kinetics \
        --label_col Label --subject_col Subject \
        --extract_pairs \
        --outdir ./simclr_eval_data/fold-2
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import SimCLRConfig
from datasets import img2tensor, imgResize, video_loader

from simclr.pretrain import build_encoder, SimCLRModel, build_view_transform

import vidaug.augmentors as va


# --------------------------------------------------------------------------- #
# Deterministic (no-augmentation) clip dataset, mirrors SimCLRVideoDataset's
# sampling but skips the two-view augmentation -- we want the plain encoded
# representation of each clip, not an augmented one, for the quality metrics.
# --------------------------------------------------------------------------- #

class CleanClipDataset(Dataset):
    def __init__(self, video_meta, config):
        """video_meta: list of dicts with keys path/label/group/split"""
        self.config = config
        self.frame_cache = {}
        sampler = va.SlidingWindowSampler(clip_len=config.clip_len, step=config.step_size)

        self.samples = []  # (path, frame_indices, label, group, split)
        for meta in video_meta:
            frames = self._get_frames(meta["path"])
            for window in sampler(len(frames)):
                self.samples.append((meta["path"], window, meta["label"], meta["group"], meta["split"]))

    def _get_frames(self, path):
        if path not in self.frame_cache:
            self.frame_cache[path] = video_loader(path)
        return self.frame_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame_indices, label, group, split = self.samples[idx]
        frames = self._get_frames(path)
        clip = [frames[i] for i in frame_indices]
        resized = [imgResize(f, target_size=self.config.frame_size) for f in clip]
        x = img2tensor(np.array(resized), self.config.repeat_chn)
        return x, label, group, split, path


class PairClipDataset(CleanClipDataset):
    """Same sampling, but returns the two SimCLR-style augmented views
    instead of the clean clip -- used only for alignment/uniformity."""
    def __init__(self, video_meta, config):
        super().__init__(video_meta, config)
        self.transform_a = build_view_transform(config, "a")
        self.transform_b = build_view_transform(config, "b")

    def _to_tensor(self, clip, augmentor):
        aug_clip = augmentor(clip)
        resized = [imgResize(f, target_size=self.config.frame_size) for f in aug_clip]
        return img2tensor(np.array(resized), self.config.repeat_chn)

    def __getitem__(self, idx):
        path, frame_indices, label, group, split = self.samples[idx]
        frames = self._get_frames(path)
        clip = [frames[i] for i in frame_indices]
        xi = self._to_tensor(clip, self.transform_a)
        xj = self._to_tensor(clip, self.transform_b)
        return xi, xj, label, group, split, path


# --------------------------------------------------------------------------- #
# CSV -> per-video metadata (label, subject/group, split)
# --------------------------------------------------------------------------- #

def resolve_column(df, requested, candidates, kind):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"--{kind}_col '{requested}' not found. Columns available: {list(df.columns)}")
        return requested
    for c in candidates:
        if c in df.columns:
            print(f"[info] no --{kind}_col given; auto-detected column '{c}' for {kind}. "
                  f"Pass --{kind}_col explicitly if this is wrong.")
            return c
    raise ValueError(
        f"Could not auto-detect a {kind} column. Pass --{kind}_col explicitly. "
        f"Columns available: {list(df.columns)}"
    )


def build_video_meta(config, fold_id, video_root, label_col, subject_col, subject_regex):
    df = pd.read_csv(config.data_csv)

    label_col = resolve_column(df, label_col, ["Label", "label", "Class", "class", "y", "target", "Diagnosis"], "label")
    if subject_col is None and subject_regex is None:
        # fall back to auto-detect a subject column; if none exists, fall back
        # to one-group-per-video (safe but coarser than true subject grouping)
        try:
            subject_col = resolve_column(df, None, ["Subject", "subject", "SubjectID", "subject_id", "PatientID", "patient_id"], "subject")
        except ValueError:
            print("[warn] no subject column found and no --subject_regex given. "
                  "Falling back to one group per VIDEO (not per true subject). "
                  "If multiple videos belong to the same subject, pass --subject_col "
                  "or --subject_regex so grouped CV doesn't leak subjects across folds.")
            subject_col = None

    df = df.drop_duplicates(subset="Video").copy()
    df["_abs_path"] = df["Video"].apply(lambda p: os.path.join(video_root, p.replace("\\", os.sep)))

    def get_group(row):
        if subject_regex is not None:
            m = re.search(subject_regex, row["Video"])
            if not m:
                raise ValueError(f"--subject_regex did not match Video value: {row['Video']}")
            return m.group(1)
        if subject_col is not None:
            return row[subject_col]
        return row["Video"]  # one group per video, safe fallback

    meta = []
    for _, row in df.iterrows():
        split = "held_out_subjects" if row["cv_fold"] == fold_id else "train_subjects"
        meta.append({
            "path": row["_abs_path"],
            "label": row[label_col],
            "group": get_group(row),
            "split": split,
        })
    return meta


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_fold_model(config, checkpoint_dir, init_mode, fold_id, proj_dim, device):
    ckpt_dir = os.path.join(checkpoint_dir, init_mode, f"fold-{fold_id}")
    full_ckpt_path = os.path.join(ckpt_dir, "simclr_full.pt")
    encoder_ckpt_path = os.path.join(ckpt_dir, "simclr_encoder.pt")

    encoder, feature_dim = build_encoder(config, from_scratch=(init_mode == "scratch"))
    model = SimCLRModel(encoder, feature_dim, proj_dim=proj_dim).to(device)

    if os.path.exists(full_ckpt_path):
        model.load_state_dict(torch.load(full_ckpt_path, map_location=device))
        print(f"[info] loaded full SimCLR model (encoder+projector) from {full_ckpt_path}")
    elif os.path.exists(encoder_ckpt_path):
        model.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
        print(f"[warn] simclr_full.pt not found; loaded encoder-only weights from {encoder_ckpt_path}. "
              f"Projector is at its Kinetics-init/random state -- do NOT trust alignment/uniformity "
              f"numbers computed from it if you pass --extract_pairs.")
    else:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}. Expected simclr_full.pt or simclr_encoder.pt.")

    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_clean_embeddings(model, video_meta, config, device, batch_size, num_workers, max_clips_per_video=None):
    dataset = CleanClipDataset(video_meta, config)
    if max_clips_per_video is not None:
        # subsample uniformly within each video to cap runtime on long videos
        rng = np.random.RandomState(0)
        by_path = {}
        for i, s in enumerate(dataset.samples):
            by_path.setdefault(s[0], []).append(i)
        keep = []
        for path, idxs in by_path.items():
            if len(idxs) > max_clips_per_video:
                idxs = list(rng.choice(idxs, max_clips_per_video, replace=False))
            keep.extend(idxs)
        dataset.samples = [dataset.samples[i] for i in sorted(keep)]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_h, all_labels, all_groups, all_splits, all_paths = [], [], [], [], []
    with torch.no_grad():
        for x, label, group, split, path in tqdm(loader, desc="extracting clean embeddings"):
            x = x.to(device)
            h = model.encoder(x)
            all_h.append(h.cpu().numpy())
            all_labels.extend(label)
            all_groups.extend(group)
            all_splits.extend(split)
            all_paths.extend(path)

    return (np.concatenate(all_h, axis=0), np.array(all_labels), np.array(all_groups),
            np.array(all_splits), np.array(all_paths))


def extract_pair_embeddings(model, video_meta, config, device, batch_size, num_workers, max_clips_per_video=None):
    dataset = PairClipDataset(video_meta, config)
    if max_clips_per_video is not None:
        rng = np.random.RandomState(0)
        by_path = {}
        for i, s in enumerate(dataset.samples):
            by_path.setdefault(s[0], []).append(i)
        keep = []
        for path, idxs in by_path.items():
            if len(idxs) > max_clips_per_video:
                idxs = list(rng.choice(idxs, max_clips_per_video, replace=False))
            keep.extend(idxs)
        dataset.samples = [dataset.samples[i] for i in sorted(keep)]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    z1_all, z2_all = [], []
    with torch.no_grad():
        for xi, xj, *_ in tqdm(loader, desc="extracting projector pair embeddings"):
            xi, xj = xi.to(device), xj.to(device)
            z1_all.append(model(xi).cpu().numpy())
            z2_all.append(model(xj).cpu().numpy())
    return np.concatenate(z1_all, axis=0), np.concatenate(z2_all, axis=0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, help="Which fold's checkpoint to load (matches --fold used in simclr_pretrain.py)")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/simclr")
    parser.add_argument("--init_mode", choices=["kinetics", "scratch"], default="kinetics")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--video_root", default=".")
    parser.add_argument("--label_col", default=None, help="CSV column with class labels. Auto-detected if omitted.")
    parser.add_argument("--subject_col", default=None, help="CSV column with subject/patient IDs, for grouped CV.")
    parser.add_argument("--subject_regex", default=None, help="Regex with one capture group to derive subject ID from the Video path, if no subject column exists.")
    parser.add_argument("--extract_pairs", action="store_true", help="Also extract two-view projector embeddings for alignment/uniformity metrics.")
    parser.add_argument("--max_clips_per_video", type=int, default=None, help="Cap clips per video to speed up extraction on long videos.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SimCLRConfig()

    video_meta = build_video_meta(config, args.fold, args.video_root, args.label_col, args.subject_col, args.subject_regex)
    n_train_videos = sum(1 for m in video_meta if m["split"] == "train_subjects")
    n_heldout_videos = sum(1 for m in video_meta if m["split"] == "held_out_subjects")
    print(f"[info] fold {args.fold}: {n_train_videos} train-subject videos, {n_heldout_videos} held-out-subject videos")

    model = load_fold_model(config, args.checkpoint_dir, args.init_mode, args.fold, args.proj_dim, device)

    h, labels, groups, splits, paths = extract_clean_embeddings(
        model, video_meta, config, device, args.batch_size, args.num_workers, args.max_clips_per_video)

    np.save(os.path.join(args.outdir, "embeddings.npy"), h)
    np.save(os.path.join(args.outdir, "labels.npy"), labels)
    np.save(os.path.join(args.outdir, "groups.npy"), groups)
    np.save(os.path.join(args.outdir, "splits.npy"), splits)
    np.save(os.path.join(args.outdir, "video_paths.npy"), paths)
    print(f"[info] saved {h.shape[0]} clip embeddings ({h.shape[1]}-d) to {args.outdir}")
    print(f"       held_out_subjects clips: {(splits == 'held_out_subjects').sum()}")
    print(f"       train_subjects clips:    {(splits == 'train_subjects').sum()}")

    if args.extract_pairs:
        z1, z2 = extract_pair_embeddings(
            model, video_meta, config, device, args.batch_size, args.num_workers, args.max_clips_per_video)
        np.save(os.path.join(args.outdir, "z_view1.npy"), z1)
        np.save(os.path.join(args.outdir, "z_view2.npy"), z2)
        print(f"[info] saved projector pair embeddings ({z1.shape[1]}-d) for alignment/uniformity")

    print("\nNext step: run python -m simclr.fold_evaluate --data_dir", args.outdir)
