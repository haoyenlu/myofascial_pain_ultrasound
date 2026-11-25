r"""
Grad-CAM for the VDE (diffusion-encoder + 3D ResNet head) video classifier,
animated as a GIF over the whole clip.

Why this differs from "normal" (2D image) Grad-CAM:
- A 2D ResNet50's last conv feature map is (C, H, W) -- one 2D heatmap.
- VDE's `head` (Resnet3DHead) produces a spatiotemporal feature map of
  shape (C, T', H', W') -- one HxW map per (downsampled) timestep. To see
  the heatmap for one specific input frame, you have to:
    1. Get the 3D CAM volume (T', H', W') -- T' may be smaller than your
       input clip length depending on the head's stride config.
    2. Upsample the whole (T', H', W') volume back to (T_in, H_in, W_in)
       with trilinear interpolation, so every input frame has its own map.
    3. Pick out the slice for the frame index you actually want to look at.

Where to hook, for VDE specifically:
  VDE's forward pass is: raw frame -> (frozen diffusion U-Net, run under
  `torch.no_grad()`) -> `encoder.conv` (small trainable conv+norm+pool
  projection) -> `head` (the 3D ResNet, Resnet3DHead) -> optional demo
  fusion -> final linear layer.

  The diffusion U-Net itself is *not hookable* for Grad-CAM at all -- it
  runs inside `torch.no_grad()` in `DiffusionEncoder.forward`, so no
  autograd graph exists there; a backward hook placed anywhere inside
  `model.encoder.diffusion` will simply never fire.

  `model.encoder.conv` *is* on the autograd graph (it's a fresh trainable
  module, applied to the diffusion features *after* the no_grad block) and
  is hookable, but it's a single conv+norm+pool, not a ResNet block.

  The actual "last ResNet layer" -- the target this script uses -- is the
  final stage of `model.head.output_3d_layer`, i.e.
  `model.head.output_3d_layer[0][-1]`: `Resnet3DHead.output_3d_layer` is
  `nn.Sequential(resnet_block_3d, avgpool, Flatten, last_linear)`, and
  `resnet_block_3d` is itself `nn.Sequential` of one stage per entry in
  `planes_3d_blocks` (currently [32, 64] -> 2 stages) -- so `[0][-1]` is
  the final (highest-channel-count) stage, directly analogous to a 2D
  ResNet's `layer4`. This indexing is robust to `planes_3d_blocks`
  changing length -- it always grabs whichever stage is last.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM3D:
    def __init__(self, model, target_layer):
        """
        model: a VDE instance (called directly on the batch dict, not
               wrapped in DataWrapper -- VDE.forward handles its own
               device placement internally).
        target_layer: the module to hook, e.g.
                       model.head.output_3d_layer[0][-1]
                       (the last stage of the 3D ResNet head -- see the
                       module docstring's "Where to hook" section for why
                       this is the right target and why the diffusion
                       U-Net itself can't be hooked at all).
        """
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, data, class_idx=None):
        """
        data: the same dict VDE.forward expects: {"clip": clip_tensor,
              "clip_key": [str], and, if config.use_demo, "demo": {...}}.
              clip_tensor shape: (1, C, T, H, W). `clip_key` is required
              because the model is in eval mode here (VDE.forward fixes
              the diffusion noise deterministically per clip when
              `self.training` is False -- see VDE.forward/train_cv.py).
        Returns:
            cam: (T_in, H_in, W_in) numpy array, normalized to [0, 1]
            prob: the model's sigmoid output for this clip (float)
        """
        self.model.zero_grad()
        output = self.model(data)  # shape (1, 1) for binary BCEWithLogits

        # For binary classification there's only one logit -- backprop that.
        # (class_idx kept for interface symmetry if you extend to multi-class)
        target = output[:, 0] if class_idx is None else output[:, class_idx]
        target.sum().backward()

        prob = torch.sigmoid(output).item()

        acts = self.activations[0]        # (C, T', H', W')
        grads = self.gradients[0]         # (C, T', H', W')

        # channel importance weights = global-average-pooled gradient,
        # same as standard 2D Grad-CAM, just averaged over T' as well as H',W'
        weights = grads.mean(dim=(1, 2, 3))          # (C,)
        cam = (weights[:, None, None, None] * acts).sum(dim=0)  # (T', H', W')
        cam = F.relu(cam)

        # upsample the whole spatiotemporal volume back to the input clip's
        # (T, H, W) so every original frame gets its own heatmap slice
        _, C_in, T_in, H_in, W_in = data["clip"].shape
        cam = cam[None, None]  # (1,1,T',H',W') for interpolate
        cam = F.interpolate(cam, size=(T_in, H_in, W_in), mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, prob


def overlay_heatmap(frame_rgb, cam_slice, alpha=0.45):
    """
    frame_rgb: (H, W, 3) uint8 RGB, OR (H, W) uint8 grayscale -- ultrasound
               frames are often single-channel; this handles either.
    cam_slice: (H, W) float in [0,1] -- the Grad-CAM for that specific frame
    """
    if frame_rgb.ndim == 2:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
    elif frame_rgb.shape[-1] == 1:
        frame_rgb = cv2.cvtColor(frame_rgb[:, :, 0], cv2.COLOR_GRAY2RGB)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_slice), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


def save_gif(frames_rgb, out_path, fps=4):
    """
    frames_rgb: list of (H, W, 3) uint8 RGB arrays, in order
    fps: playback speed -- converted to per-frame duration in ms
    """
    images = [Image.fromarray(f) for f in frames_rgb]
    duration_ms = int(round(1000 / fps))
    images[0].save(out_path, save_all=True, append_images=images[1:],
                    duration=duration_ms, loop=0)


def visualize_clip(model, data, raw_frames, target_layer=None, out_path=None, fps=4, alpha=0.45):
    """
    Same Grad-CAM computation as visualize_frame, but overlays EVERY frame
    in the clip (not just one) and saves the sequence as an animated GIF.

    raw_frames: (T, H, W, 3) or (T, H, W) -- same requirement as
                visualize_frame: these must be the exact frames (same
                indices, same resize) that went into `data["clip"]`.
    target_layer: module to hook. Defaults to VDE's last ResNet-head
                  stage (`model.head.output_3d_layer[0][-1]`) -- see the
                  module docstring's "Where to hook" section. Override if
                  you want e.g. `model.encoder.conv` instead (the small
                  trainable projection before the head -- still on the
                  autograd graph, just not a "ResNet layer").
    fps: playback speed of the output GIF.
    """
    if target_layer is None:
        if hasattr(model.head, 'conv_layers'):
            target_layer = model.head.conv_layers
        else:
            target_layer = model.head.output_3d_layer[0][-1]
    gradcam = GradCAM3D(model, target_layer)
    cam, prob = gradcam(data)  # cam: (T, H, W), one heatmap per input frame
    overlays = []
    for t in range(cam.shape[0]):
        cam_slice = cam[t]
        frame = raw_frames[t]
        if frame.shape[:2] != cam_slice.shape:
            cam_slice = cv2.resize(cam_slice, (frame.shape[1], frame.shape[0]))
        overlays.append(overlay_heatmap(frame, cam_slice, alpha=alpha))

    if out_path:
        save_gif(overlays, out_path, fps=fps)
        print(f"Saved Grad-CAM GIF ({len(overlays)} frames @ {fps} fps) to {out_path} "
              f"(predicted prob={prob:.3f})")

    return overlays, cam, prob


def load_clip_from_row(row, config, video_root=".", age_scaler=None, ppt_scaler=None):
    """
    Read one video straight from a dataframe row (no Dataset/DataLoader
    involved), sample a single clip's worth of frames, and produce:
      - raw_frames: (T, H, W, 3) uint8 RGB, for a human-viewable overlay
      - clip_tensor: (1, C, T, H, W), the exact tensor the model expects
      - clip_key: str -- VDE requires this in eval mode (fixes the
                  diffusion noise deterministically per clip; see
                  train_cv.py's VideoDataset/VDE.forward). Built the same
                  way as VideoDataset does: f"{video_id}:{start}-{end}".
      - demo: dict of {"sex","age","ppt"} tensors, each with a batch dim
              of 1 -- only built if `config.use_demo`. `age_scaler` and
              `ppt_scaler` MUST be the scaler fit on this fold's TRAIN
              split (see run_gradcam) -- fitting a fresh scaler on just
              this one row, or skipping scaling entirely, would feed VDE
              demographic features on the wrong scale relative to what it
              was trained on. Pass None for both if config.use_demo is
              False (they're unused in that case).

    This mirrors what VideoDataset does internally, just done directly and
    deterministically (center clip, no augmentation) so the frames you
    overlay on are guaranteed to be the same ones the model actually saw.
    """
    from datasets import img2tensor, imgResize, video_loader
    from pathlib import Path

    video_path = os.path.join(video_root, row["Video"].replace("\\", os.sep))
    frames = video_loader(video_path)  # full decoded video, list/array of frames

    n_frames = len(frames)
    clip_len = config.clip_len

    # deterministic center clip (no sliding-window sampling here -- we just
    # want *one* representative clip to visualize). Falls back to the whole
    # video, repeated/padded, if it's shorter than clip_len.
    if n_frames >= clip_len:
        start = (n_frames - clip_len) // 2
        frame_indices = list(range(start, start + clip_len))
    else:
        # pad by repeating the last frame -- keep this simple and explicit
        # rather than reusing the sampler's pingpong fallback
        frame_indices = list(range(n_frames)) + [n_frames - 1] * (clip_len - n_frames)

    raw_clip = [frames[i] for i in frame_indices]
    resized = [imgResize(f, target_size=config.frame_size) for f in raw_clip]
    raw_frames = np.stack(resized).astype(np.uint8)  # (T, H, W, 3), uint8, for overlay

    clip_tensor = img2tensor(np.array(resized), config.repeat_chn)  # same call VideoDataset makes
    clip_tensor = clip_tensor.unsqueeze(0)  # add batch dim -> (1, C, T, H, W)

    video_id = Path(row["Video"]).stem
    clip_key = f"{video_id}:{frame_indices[0]}-{frame_indices[-1]}"

    demo = None
    if getattr(config, "use_demo", False):
        assert age_scaler is not None and ppt_scaler is not None, \
            "config.use_demo is True but no age_scaler/ppt_scaler given -- " \
            "see run_gradcam for how to fit them on this fold's train split"
        sex_label = 0 if row["Sex"] == "M" else 1
        age_std = age_scaler.transform([[row["Age"]]])[0]
        ppt_std = ppt_scaler.transform([[row["PPT_PRE_AVG"], row["PPT_POST_AVG"]]])[0]
        demo = {
            "sex": torch.tensor([[sex_label]], dtype=torch.float32),
            "age": torch.tensor([age_std], dtype=torch.float32),
            "ppt": torch.tensor([ppt_std], dtype=torch.float32),
        }

    return raw_frames, clip_tensor, clip_key, demo


def load_model_for_fold(config, fold_idx, device, checkpoint_name="best_combined_model.pth"):
    """
    Load the trained VDE checkpoint for one fold. Raises FileNotFoundError
    (caller's job to catch this if batch-processing multiple folds where
    some may not have finished training yet -- e.g. this config's
    `train_folds = [2, 4]` means folds 1/3/5 might not have a matching
    checkpoint at all right now).

    fold_idx matters for more than just the checkpoint path: get_vde_model
    also uses it to select the matching per-fold diffusion checkpoint
    (config.diffusion_fold_checkpoint), same as during training.
    """
    from train_cv import get_vde_model

    checkpoint_path = os.path.join(config.checkpoint_dir, f"fold-{fold_idx}", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- check config.checkpoint_dir "
            f"and that fold {fold_idx} finished training."
        )
    model = get_vde_model(config, fold_id=fold_idx, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def fit_demo_scalers(df, fold_idx, config):
    """
    Fit age/PPT scalers on this fold's TRAIN split only (cv_fold !=
    fold_idx), exactly mirroring train_cv.py's main() -- never refit on
    a single visualized row, or on the whole df, or demo features end up
    on a different scale than what the model was actually trained on.
    Returns (None, None) if config.use_demo is False (current default in
    this config).
    """
    if not getattr(config, "use_demo", False):
        return None, None
    from sklearn.preprocessing import StandardScaler

    train_df = df[df["cv_fold"] != fold_idx]
    age_scaler = StandardScaler().fit(train_df[["Age"]])
    ppt_scaler = StandardScaler().fit(train_df[["PPT_PRE_AVG", "PPT_POST_AVG"]])
    return age_scaler, ppt_scaler


def visualize_subject_video(model, row, config, device, video_root, fps, out_path,
                             age_scaler=None, ppt_scaler=None):
    """
    One clip, start to finish: load frames from `row`, build the model's
    input dict, run Grad-CAM, save the overlay GIF. Shared by both
    run_gradcam (single video) and run_gradcam_batch (every subject,
    every fold) so there's exactly one place this logic lives.

    Returns (overlays, cam, prob) -- same as visualize_clip.
    """
    raw_frames, clip_tensor, clip_key, demo = load_clip_from_row(
        row, config, video_root=video_root, age_scaler=age_scaler, ppt_scaler=ppt_scaler)

    single = {"clip": clip_tensor.to(device), "clip_key": [clip_key]}
    if demo is not None:
        single["demo"] = {k: v.to(device) for k, v in demo.items()}

    return visualize_clip(model, single, raw_frames, out_path=out_path, fps=fps)


def run_gradcam(fold_idx, row_selector=0, fps=4, video_root=".",
                 checkpoint_name="best_combined_model.pth", out_dir="."):
    """
    fold_idx: which fold's checkpoint + validation video to use (1-5)
    row_selector: either an int (positional row index within that fold's
                  validation rows) or a string (a specific Study ID to look
                  up within that fold)
    fps: playback speed of the output GIF
    """
    import sys
    sys.path.append(".")
    from config import VDEConfig
    import pandas as pd

    config = VDEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_for_fold(config, fold_idx, device, checkpoint_name)

    # --- pick a video from THIS fold's validation split ---
    # (cv_fold == fold_idx means "held out for this fold" -- i.e. exactly the
    # subjects this checkpoint was validated on, never trained on)
    df = pd.read_csv(config.data_csv)
    val_df = df[df["cv_fold"] == fold_idx].reset_index(drop=True)
    if len(val_df) == 0:
        raise ValueError(f"No rows found with cv_fold == {fold_idx} in {config.data_csv}")

    if isinstance(row_selector, str):
        matches = val_df[val_df["Study ID"] == row_selector]
        if len(matches) == 0:
            raise ValueError(f"Study ID '{row_selector}' not found in fold {fold_idx}'s validation set. "
                              f"Available IDs: {val_df['Study ID'].unique()}")
        row = matches.iloc[0]
    else:
        row = val_df.iloc[row_selector]

    print(f"Fold {fold_idx} | Subject: {row['Study ID']} | Side: {row.get('Side','?')} | "
          f"true label: {row['label']} | video: {row['Video']}")

    age_scaler, ppt_scaler = fit_demo_scalers(df, fold_idx, config)

    video_id_stem = os.path.splitext(os.path.basename(row["Video"]))[0]
    out_path = os.path.join(out_dir, f"gradcam_fold{fold_idx}_{row['Study ID']}_{video_id_stem}.gif")
    overlays, cam, prob = visualize_subject_video(model, row, config, device, video_root, fps, out_path,
                                                    age_scaler, ppt_scaler)
    print(f"Predicted prob: {prob:.3f}  |  true label: {row['label']}")
    return overlays, cam, prob


def run_gradcam_batch(fold_range=None, videos_per_subject=1, checkpoint_name="best_combined_model.pth",
                       video_root=".", out_dir="./gradcam_outputs", fps=4):
    """
    Generate a Grad-CAM GIF for every subject, across every fold, in one
    call.

    fold_range: iterable of fold indices to process. Defaults to
                range(1, config.n_fold+1) -- i.e. all 5 folds. Folds
                without a finished checkpoint (e.g. right now only 2/4
                have one, per this config's `train_folds = [2, 4]`) are
                skipped with a warning, not fatal -- the rest of the
                batch still runs.
    videos_per_subject: how many of a subject's videos to visualize.
                Default 1 (their first video in that fold's validation
                split, by row order) -- this is "every SUBJECT", not
                "every video"; some subjects have many videos,
                so doing all of them for everyone would produce ~300
                GIFs rather than ~38. Pass "all" (or None) to instead do
                every video for every subject.
    out_dir: GIFs are written to {out_dir}/fold-{fold_idx}/{subject}_{video_id}.gif.
             A manifest CSV (one row per attempted clip, with the
             predicted probability, true label, and pass/fail status) is
             written to {out_dir}/manifest.csv at the end -- useful for
             sorting/filtering which clips to actually look at without
             opening every GIF.

    Returns the manifest as a list of dicts (also written to CSV).
    """
    import sys
    sys.path.append(".")
    from config import VDEConfig
    import pandas as pd
    from pathlib import Path

    config = VDEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fold_range is None:
        fold_range = range(1, config.n_fold + 1)

    df = pd.read_csv(config.data_csv)
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    n_ok, n_failed, n_skipped_folds = 0, 0, 0

    for fold_idx in fold_range:
        try:
            model = load_model_for_fold(config, fold_idx, device, checkpoint_name)
        except FileNotFoundError as e:
            print(f"[fold {fold_idx}] SKIPPED -- {e}")
            n_skipped_folds += 1
            continue

        val_df = df[df["cv_fold"] == fold_idx].reset_index(drop=True)
        if len(val_df) == 0:
            print(f"[fold {fold_idx}] SKIPPED -- no rows with cv_fold == {fold_idx}")
            n_skipped_folds += 1
            continue

        age_scaler, ppt_scaler = fit_demo_scalers(df, fold_idx, config)

        fold_out_dir = os.path.join(out_dir, f"fold-{fold_idx}")
        os.makedirs(fold_out_dir, exist_ok=True)

        subjects = sorted(val_df["Study ID"].unique())
        print(f"[fold {fold_idx}] {len(subjects)} subjects in validation split: {subjects}")

        for subject in subjects:
            subject_rows = val_df[val_df["Study ID"] == subject].reset_index(drop=True)
            if videos_per_subject in (None, "all"):
                n_take = len(subject_rows)
            else:
                n_take = min(videos_per_subject, len(subject_rows))

            for i in range(n_take):
                row = subject_rows.iloc[i]
                video_id = Path(row["Video"]).stem
                out_path = os.path.join(fold_out_dir, f"{subject}_{video_id}.gif")

                try:
                    overlays, cam, prob = visualize_subject_video(model, row, config, device, video_root, fps,
                                                                     out_path, age_scaler, ppt_scaler)
                    print(f"[fold {fold_idx}] {subject} / {video_id}: prob={prob:.3f} "
                          f"label={row['label']}  -> {out_path}")
                    manifest.append(dict(fold=fold_idx, subject=subject, video_id=video_id,
                                          true_label=row["label"], predicted_prob=prob,
                                          gif_path=out_path, status="ok"))
                    n_ok += 1
                except Exception as e:
                    print(f"[fold {fold_idx}] {subject} / {video_id}: FAILED -- {e}")
                    manifest.append(dict(fold=fold_idx, subject=subject, video_id=video_id,
                                          true_label=row.get("label"), predicted_prob=None,
                                          gif_path=None, status=f"FAILED: {e}"))
                    n_failed += 1

    manifest_path = os.path.join(out_dir, "manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"\nDone. {n_ok} GIFs written, {n_failed} failed, {n_skipped_folds} folds skipped "
          f"(missing checkpoint). Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM GIF(s) for VDE fold checkpoint(s)")
    parser.add_argument("--all_subjects", action="store_true",
                         help="Batch mode: generate a GIF for every subject, across every fold "
                              "(or just --fold, if given). Ignores --row.")
    parser.add_argument("--fold", type=int, nargs="+", default=None,
                         help="Fold index/indices (1-5). Required in single-video mode. "
                              "In --all_subjects mode, defaults to all folds (1..n_fold) if omitted.")
    parser.add_argument("--row", default="0",
                         help="[single-video mode only] Row selector within that fold's val set: "
                              "an integer position (e.g. '0') or a Study ID string (e.g. 'SUBJ01')")
    parser.add_argument("--videos_per_subject", default="1",
                         help="[--all_subjects mode only] How many videos per subject to visualize. "
                              "An integer, or 'all' for every video every subject has.")
    parser.add_argument("--fps", type=int, default=4, help="Playback speed of the output GIF")
    parser.add_argument("--video_root", default=".", help="Root directory the CSV's Video paths are relative to")
    parser.add_argument("--checkpoint_name", default="best_combined_model.pth")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    if args.all_subjects:
        videos_per_subject = args.videos_per_subject
        if videos_per_subject != "all":
            videos_per_subject = int(videos_per_subject)
        run_gradcam_batch(
            fold_range=args.fold,  # None -> all folds, inside run_gradcam_batch
            videos_per_subject=videos_per_subject,
            checkpoint_name=args.checkpoint_name,
            video_root=args.video_root,
            out_dir=args.out_dir,
            fps=args.fps,
        )
    else:
        if args.fold is None:
            parser.error("--fold is required in single-video mode (or use --all_subjects)")
        row_selector = int(args.row) if args.row.isdigit() else args.row
        run_gradcam(
            fold_idx=args.fold[0],
            row_selector=row_selector,
            fps=args.fps,
            video_root=args.video_root,
            checkpoint_name=args.checkpoint_name,
            out_dir=args.out_dir,
        )
