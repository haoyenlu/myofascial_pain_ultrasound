r"""
Grad-CAM for the VideoMAE (transformer) video classifier, animated as a GIF
over the whole clip -- same output format as gradCam3d.py, different
plumbing because VideoMAE is a ViT, not a conv net.

Why this differs from gradCam3d.py's 3D-ResNet Grad-CAM:
- The 3D ResNet's last conv layer gives you a (C, T', H', W') feature map --
  spatial/temporal structure is already there, baked into the tensor shape.
- VideoMAE's encoder blocks output a *flat sequence of tokens*:
  (B, N, D), where each of the N tokens is one spatiotemporal "tubelet"
  patch and D is the hidden size. There's no CLS token in HF's
  VideoMAEModel (pooling is done by mean-pooling all patch tokens through
  fc_norm), so N is *exactly* T' * H' * W' with:
      T' = config.num_frames // config.tubelet_size
      H' = config.image_size  // config.patch_size
      W' = config.image_size  // config.patch_size
  To get something you can overlay on frames, you have to:
    1. Hook a transformer block (not a conv layer) to grab its output
       tokens (B, N, D) and the gradient flowing back into them.
    2. Standard Grad-CAM weighting, just done over the token axis instead
       of over (T,H,W): weight per hidden channel d = mean gradient over
       all N tokens. Then per-token CAM value = ReLU(sum_d weight_d *
       activation[token, d]) -- a dot product per token instead of a
       per-pixel weighted sum over channels.
    3. Reshape the resulting (N,) vector back into (T', H', W') -- this
       only works because VideoMAE's patch order is (temporal, then row,
       then column), matching how the embeddings were built.
    4. From here it's identical to the ResNet version: trilinear-upsample
       (T', H', W') -> (T_in, H_in, W_in) and slice out the frame you want.

Where to hook: target the LAST transformer block (encoder.layer[-1]) by
default. That's the deepest, most task-specific block, closest in spirit to
"layer4" in the ResNet version (freezing schemes here also always leave the
head + last blocks trainable -- see VMAEConfig / num_frozen_blocks in
videomae_finetune.py). Pass `target_block_idx` if you want a different one
(e.g. the first *trainable* block instead of the very last one).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

# reuse the frame-overlay / GIF-writing helpers verbatim -- they only care
# about (H, W) heatmaps and RGB frames, nothing ResNet-specific
from utils.gradcam3d import overlay_heatmap, save_gif


class GradCAMVideoMAE:
    def __init__(self, model, target_block_idx=-1):
        """
        model: a VideoMAEForVideoClassification (the thing build_model() /
               from_pretrained() returns)
        target_block_idx: index into model.videomae.encoder.layer to hook.
                           Defaults to -1 (the last block).
        """
        self.model = model
        self.activations = None
        self.gradients = None
        print(f"Video MAE N_layer={len(model.videomae.encoder.layer)}")
        target_layer = model.videomae.encoder.layer[target_block_idx]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        # VideoMAELayer returns a tuple: (hidden_states,) or
        # (hidden_states, attn_probs) if output_attentions=True
        hs = output[0] if isinstance(output, tuple) else output
        self.activations = hs.detach()  # (B, N, D)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (B, N, D)

    def __call__(self, pixel_values, class_idx=None):
        """
        pixel_values: (1, T, C, H, W) -- VideoMAE's expected input order
                       (batch, frames, channels, height, width), NOT the
                       (B, C, T, H, W) order the 3D ResNet uses.
        Returns:
            cam: (T_in, H_in, W_in) numpy array, normalized to [0, 1]
            prob: the model's sigmoid output for this clip (float)
        """
        self.model.zero_grad()
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, 1) for binary, num_labels=1

        target = logits[:, 0] if class_idx is None else logits[:, class_idx]
        target.sum().backward()

        prob = torch.sigmoid(logits).item()

        acts = self.activations[0]   # (N, D)
        grads = self.gradients[0]    # (N, D)

        # channel importance weights = gradient averaged over all tokens
        # (the token axis is the transformer's stand-in for "all spatial
        # positions" in the conv case)
        weights = grads.mean(dim=0)                       # (D,)
        cam = (acts * weights[None, :]).sum(dim=1)         # (N,)
        cam = F.relu(cam)

        cfg = self.model.config
        T_prime = cfg.num_frames // cfg.tubelet_size
        H_prime = cfg.image_size // cfg.patch_size
        W_prime = cfg.image_size // cfg.patch_size
        expected_n = T_prime * H_prime * W_prime
        if cam.numel() != expected_n:
            raise ValueError(
                f"Token count {cam.numel()} != T'*H'*W' ({expected_n}) -- "
                f"check tubelet_size/patch_size against the actual model config "
                f"({cfg.tubelet_size=}, {cfg.patch_size=}, {cfg.num_frames=}, {cfg.image_size=})."
            )

        # tokens are ordered (temporal, row, column) by VideoMAE's patch
        # embedding, so this reshape recovers real spatiotemporal structure
        cam = cam.reshape(1, 1, T_prime, H_prime, W_prime)

        _, T_in, _, H_in, W_in = pixel_values.shape  # (B, T, C, H, W)
        cam = F.interpolate(cam, size=(T_in, H_in, W_in), mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, prob


def visualize_clip_videomae(model, pixel_values, raw_frames, out_path=None,
                             fps=4, alpha=0.45, target_block_idx=-1):
    """
    Same idea as gradCam3d.visualize_clip, just fed a VideoMAE model +
    (1, T, C, H, W) pixel_values instead of a ResNet + {"clip": ...} dict.

    raw_frames: (T, H, W, 3) or (T, H, W) uint8 -- must be the exact frames
                (same indices, same resize) that were normalized into
                pixel_values, same requirement as the ResNet version.
    """
    gradcam = GradCAMVideoMAE(model, target_block_idx=target_block_idx)
    cam, prob = gradcam(pixel_values)  # cam: (T, H, W)

    overlays = []
    for t in range(cam.shape[0]):
        cam_slice = cam[t]
        frame = raw_frames[t]
        if frame.shape[:2] != cam_slice.shape:
            import cv2
            cam_slice = cv2.resize(cam_slice, (frame.shape[1], frame.shape[0]))
        overlays.append(overlay_heatmap(frame, cam_slice, alpha=alpha))

    if out_path:
        save_gif(overlays, out_path, fps=fps)
        print(f"Saved Grad-CAM GIF ({len(overlays)} frames @ {fps} fps) to {out_path} "
              f"(predicted prob={prob:.3f})")

    return overlays, cam, prob


def load_clip_from_row_videomae(row, config, num_frames, image_size, video_root="."):
    """
    VideoMAE analog of gradCam3d.load_clip_from_row: read one video, sample
    a single deterministic center clip, and produce both:
      - raw_frames: (T, H, W, 3) uint8 RGB, for the human-viewable overlay
      - pixel_values: (1, T, C, H, W) float tensor, VideoMAE's expected
        input order and normalization

    Mirrors VideoMAEClassificationDataset (no sliding window / augmentation
    here -- one representative center clip, deterministic).
    """
    from datasets import imgResize, video_loader
    from videomae.pretrain import frame_to_normalized_tensor

    print(f"Using N_frames {num_frames}, Frame Size {image_size}")
    video_path = os.path.join(video_root, row["Video"].replace("\\", os.sep))
    frames = video_loader(video_path)

    n_frames = len(frames)

    clip_len = num_frames # VideoMAEConfig / model.config.num_frames

    if n_frames >= clip_len:
        start = (n_frames - clip_len) // 2
        frame_indices = list(range(start, start + clip_len))
    else:
        frame_indices = list(range(n_frames)) + [n_frames - 1] * (clip_len - n_frames)

    raw_clip = [frames[i] for i in frame_indices]
    resized = [imgResize(f, target_size=image_size) for f in raw_clip]
    raw_frames = np.stack(resized).astype(np.uint8)  # (T, H, W, 3), for overlay

    pixel_values = frame_to_normalized_tensor(resized, image_size)  # (T, C, H, W)
    pixel_values = pixel_values.unsqueeze(0)  # -> (1, T, C, H, W)

    return raw_frames, pixel_values


def load_videomae_checkpoint(fold_idx, init_mode, checkpoint_subdir, config, device):
    """
    Shared checkpoint-loading logic, factored out of run_gradcam_videomae so
    both the single-video runner and the all-subjects batch runner load the
    model exactly once and the same way.
    """
    from transformers import VideoMAEForVideoClassification

    checkpoint_path = os.path.join(config.checkpoint_dir, init_mode, checkpoint_subdir,
                                    f"fold-{fold_idx}", 'best_combined_model')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- check config.checkpoint_dir, "
            f"the init_mode/frozen-blocks subfolder your run used, and that fold "
            f"{fold_idx} finished finetuning."
        )
    model = VideoMAEForVideoClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def run_gradcam_videomae(fold_idx, row_selector=0, fps=4, video_root=".",
                          init_mode='scratch', 
                          checkpoint_subdir="best_combined_model", out_dir=".",
                          target_block_idx=-1):
    """
    fold_idx: which fold's checkpoint + validation video to use (1-5)
    row_selector: int (positional row index in that fold's val rows) or a
                  Study ID string
    checkpoint_subdir: the folder train_fold() saved via model.save_pretrained
                        (default "best_combined_model", matching
                        videomae_finetune.py's train_fold)
    """
    import sys
    sys.path.append(".")
    from config import VMAEConfig
    import pandas as pd

    config = VMAEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_videomae_checkpoint(fold_idx, init_mode, checkpoint_subdir, config, device)

    df = pd.read_csv(config.data_csv)
    val_df = df[df["cv_fold"] == fold_idx].reset_index(drop=True)
    if len(val_df) == 0:
        raise ValueError(f"No rows found with cv_fold == {fold_idx} in {config.data_csv}")

    if isinstance(row_selector, str):
        matches = val_df[val_df["Study ID"] == row_selector]
        if len(matches) == 0:
            raise ValueError(f"Study ID '{row_selector}' not found in fold {fold_idx}'s validation set, available ID {val_df['Study ID'].unique()}")
        row = matches.iloc[0]
    else:
        row = val_df.iloc[row_selector]

    print(f"Fold {fold_idx} | Subject: {row['Study ID']} | Side: {row.get('Side', '?')} | "
          f"true label: {row['label']} | video: {row['Video']}")
    
    num_frames = model.config.num_frames
    image_size = model.config.image_size

    raw_frames, pixel_values = load_clip_from_row_videomae(row, config, num_frames=num_frames, image_size=image_size, video_root=video_root)
    pixel_values = pixel_values.to(device)

    out_path = os.path.join(out_dir, f"gradcam_videomae_fold{fold_idx}_{row['Study ID']}.gif")
    overlays, cam, prob = visualize_clip_videomae(
        model, pixel_values, raw_frames, out_path=out_path, fps=fps,
        target_block_idx=target_block_idx,
    )
    print(f"Predicted prob: {prob:.3f}  |  true label: {row['label']}")
    return overlays, cam, prob


def run_gradcam_videomae_all_subjects(fold_idx, fps=4, video_root=".",
                                       init_mode='scratch',
                                       checkpoint_subdir="best_combined_model", out_dir=".",
                                       target_block_idx=-1, videos_per_subject="first"):
    """
    Same as run_gradcam_videomae, but loops over EVERY subject in that
    fold's validation split and generates one Grad-CAM GIF per subject,
    instead of requiring a row_selector for a single video.

    videos_per_subject: how to pick a video when a subject has more than
                         one row in the fold's val set (e.g. pre/post,
                         left/right side):
        "first" -- just take the first matching row (deterministic, based
                   on the CSV's row order) [default]
        "all"   -- generate one GIF per row (i.e. every video for every
                   subject, not just one video per subject)

    Returns a list of dicts: [{"subject": ..., "video": ..., "label": ...,
                                "prob": ..., "out_path": ...}, ...]
    for every subject/video processed, plus any that failed (with an
    "error" key instead of "prob") so one bad video doesn't kill the run.
    """
    import sys
    sys.path.append(".")
    from config import VMAEConfig
    import pandas as pd

    config = VMAEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_videomae_checkpoint(fold_idx, init_mode, checkpoint_subdir, config, device)
    num_frames = model.config.num_frames
    image_size = model.config.image_size

    df = pd.read_csv(config.data_csv)
    val_df = df[df["cv_fold"] == fold_idx].reset_index(drop=True)
    if len(val_df) == 0:
        raise ValueError(f"No rows found with cv_fold == {fold_idx} in {config.data_csv}")

    subjects = val_df["Study ID"].unique()
    print(f"Fold {fold_idx}: {len(subjects)} subject(s) in validation split -> "
          f"generating Grad-CAM GIF(s) for {'all videos' if videos_per_subject == 'all' else 'one video'} each")

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for subject in subjects:
        subject_rows = val_df[val_df["Study ID"] == subject]
        if videos_per_subject == "all":
            rows_to_process = [row._asdict() for row in subject_rows.itertuples(index=False)]
        else:
            rows_to_process = [subject_rows.iloc[0]]

        for i, row in enumerate(rows_to_process):
            video_name = os.path.splitext(os.path.basename(str(row["Video"])))[0]
            suffix = f"_{video_name}" if videos_per_subject == "all" and len(rows_to_process) > 1 else ""

            print(f"\n--- Fold {fold_idx} | Subject: {subject} | Side: {row.get('Side', '?')} | "
                  f"true label: {row['label']} | video: {row['Video']} ---")

            try:
                raw_frames, pixel_values = load_clip_from_row_videomae(
                    row, config, num_frames=num_frames, image_size=image_size, video_root=video_root
                )
                pixel_values = pixel_values.to(device)

                out_path = os.path.join(out_dir, f"gradcam_videomae_fold{fold_idx}_{subject}{suffix}.gif")
                overlays, cam, prob = visualize_clip_videomae(
                    model, pixel_values, raw_frames, out_path=out_path, fps=fps,
                    target_block_idx=target_block_idx,
                )
                print(f"Predicted prob: {prob:.3f}  |  true label: {row['label']}")
                results.append({
                    "subject": subject, "video": row["Video"], "label": row["label"],
                    "prob": prob, "out_path": out_path,
                })
            except Exception as e:
                print(f"!! Failed on subject {subject} (video={row['Video']}): {e}")
                results.append({
                    "subject": subject, "video": row["Video"], "label": row["label"],
                    "error": str(e),
                })

    n_ok = sum(1 for r in results if "prob" in r)
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok} GIF(s) generated, {n_fail} failed, saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM GIF for a specific fold's VideoMAE checkpoint + video")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (1-5)")
    parser.add_argument("--row", default="0",
                         help="Row selector within that fold's val set: an integer position "
                              "(e.g. '0') or a Study ID string (e.g. 'SUBJ01')")
    parser.add_argument("--fps", type=int, default=4, help="Playback speed of the output GIF")
    parser.add_argument("--video_root", default=".", help="Root directory the CSV's Video paths are relative to")
    parser.add_argument("--checkpoint_subdir", default="best_combined_model",
                         help="Subfolder under fold-<n>/ that model.save_pretrained wrote to")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--target_block_idx", type=int, default=-1,
                         help="Which transformer block to hook (default: last block, -1)")

    parser.add_argument("--from_scratch", action="store_true",
                         help="Finetune from the from-scratch-pretrained encoders instead of the "
                              "Kinetics-init ones -- must match how you ran pretrain_videomae.py")

    parser.add_argument("--all_subjects", action="store_true",
                         help="Ignore --row and generate a Grad-CAM GIF for every subject in this "
                              "fold's validation split instead of just one video")
    parser.add_argument("--videos_per_subject", choices=["first", "all"], default="first",
                         help="Only used with --all_subjects. 'first' (default) makes one GIF per "
                              "subject using their first video; 'all' makes one GIF per video, "
                              "covering every video for every subject.")
    args = parser.parse_args()

    init_mode = "scratch" if args.from_scratch else "kinetics"

    if args.all_subjects:
        run_gradcam_videomae_all_subjects(
            fold_idx=args.fold,
            fps=args.fps,
            video_root=args.video_root,
            init_mode=init_mode,
            checkpoint_subdir=args.checkpoint_subdir,
            out_dir=args.out_dir,
            target_block_idx=args.target_block_idx,
            videos_per_subject=args.videos_per_subject,
        )
    else:
        row_selector = int(args.row) if args.row.isdigit() else args.row
        run_gradcam_videomae(
            fold_idx=args.fold,
            row_selector=row_selector,
            fps=args.fps,
            video_root=args.video_root,
            init_mode=init_mode,
            checkpoint_subdir=args.checkpoint_subdir,
            out_dir=args.out_dir,
            target_block_idx=args.target_block_idx,
        )
