r"""
Grad-CAM for the 3D ResNet50 video classifier, animated as a GIF over the
whole clip.

Why this differs from "normal" (2D image) Grad-CAM:
- A 2D ResNet50's last conv feature map is (C, H, W) -- one 2D heatmap.
- This model's `resnet50(**resnet_param)` is a 3D (video) ResNet: its last
  conv feature map is (C, T, H, W) -- a *spatiotemporal* heatmap, one HxW map
  per (downsampled) timestep. To see the heatmap for one specific input
  frame, you have to:
    1. Get the 3D CAM volume (T', H', W') -- T' is smaller than your input
       clip length because of temporal downsampling (strided convs/pools).
    2. Upsample the whole (T', H', W') volume back to (T_in, H_in, W_in)
       with trilinear interpolation, so every input frame has its own map.
    3. Pick out the slice for the frame index you actually want to look at.

Where to hook: `freeze_layers = (conv1., layer1., layer2., layer3.)` means
layer4 is the only trainable conv block -- it's also the right Grad-CAM
target, since it holds the most class-specific features (the frozen layers
mostly encode generic pretrained motion/texture, not your Latent/Healthy
signal).
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM3D:
    def __init__(self, model, target_layers):
        """
        model: your DataWrapper-wrapped resnet (the thing get_model() returns)
        target_layers: a single module, OR a list of modules to hook, e.g.
                       [model.model.layer3, model.model.layer4].

                       Why more than one layer can matter here: only
                       layer4 is trainable (layer1-3 are frozen, per
                       config.freeze_layers) -- but layer4 is also
                       typically where this 3D ResNet has downsampled the
                       TEMPORAL dimension the most (T' at layer4 is often
                       much smaller than at layer2/layer3 -- print
                       `acts.shape` for each candidate layer on one
                       forward pass if you want to confirm the exact
                       numbers for this architecture). A Grad-CAM built
                       from layer4 alone can end up nearly temporally
                       flat: interpolating a CAM volume with very small T'
                       back up to the full input clip length mostly just
                       repeats the same few timesteps' maps across many
                       frames, so the resulting GIF shows little real
                       per-frame variation despite technically having
                       "one map per frame."

                       Passing multiple layers computes an independently-
                       normalized CAM at EACH one (after upsampling every
                       layer's volume to the same full clip shape), then
                       averages them: layer4 keeps the actual class-
                       discriminative signal (the only layer that got
                       gradient updates during training), while an
                       earlier, frozen layer like layer3 contributes real
                       frame-to-frame temporal variation that the coarser
                       class-specific layer can't provide on its own.
        """
        if not isinstance(target_layers, (list, tuple)):
            target_layers = [target_layers]
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        for layer in target_layers:
            layer.register_forward_hook(self._make_save_activation(layer))
            layer.register_full_backward_hook(self._make_save_gradient(layer))

    def _make_save_activation(self, layer):
        def hook(module, inp, output):
            self.activations[layer] = output.detach()
        return hook

    def _make_save_gradient(self, layer):
        def hook(module, grad_input, grad_output):
            self.gradients[layer] = grad_output[0].detach()
        return hook

    def __call__(self, data, class_idx=None):
        """
        data: the same dict your DataWrapper expects, e.g. {"clip": clip_tensor}
              clip_tensor shape: (1, C, T, H, W)
        Returns:
            cam: (T_in, H_in, W_in) numpy array, normalized to [0, 1] --
                 averaged across all target_layers if more than one was given.
            prob: the model's sigmoid output for this clip (float)
        """
        self.model.zero_grad()

        # CRITICAL for hooking any of layer1-3: their weights are frozen
        # (requires_grad=False), and the raw input clip tensor doesn't
        # require grad by default either. With BOTH of those false,
        # autograd never builds a graph through those layers at all --
        # their backward hooks would simply never fire (no error raised;
        # self.gradients would just stay empty for that layer). Marking
        # the INPUT as requiring grad re-establishes a valid gradient
        # path through every layer downstream of it, frozen or not,
        # independent of which layers' *weights* happen to be trainable.
        data["clip"].requires_grad_(True)

        output = self.model(data)  # shape (1, 1) for binary BCEWithLogits

        # For binary classification there's only one logit -- backprop that.
        # (class_idx kept for interface symmetry if you extend to multi-class)
        target = output[:, 0] if class_idx is None else output[:, class_idx]
        target.sum().backward()

        prob = torch.sigmoid(output).item()

        _, C_in, T_in, H_in, W_in = data["clip"].shape

        per_layer_cams = []
        for layer in self.target_layers:
            acts = self.activations[layer][0]   # (C, T', H', W')
            grads = self.gradients[layer][0]     # (C, T', H', W')

            # channel importance weights = global-average-pooled gradient,
            # same as standard 2D Grad-CAM, just averaged over T' as well as H',W'
            weights = grads.mean(dim=(1, 2, 3))          # (C,)
            cam = (weights[:, None, None, None] * acts).sum(dim=0)  # (T', H', W')
            cam = F.relu(cam)

            # upsample THIS layer's spatiotemporal volume back to the input
            # clip's (T, H, W) so every original frame gets its own slice
            cam = cam[None, None]  # (1,1,T',H',W') for interpolate
            cam = F.interpolate(cam, size=(T_in, H_in, W_in), mode="trilinear", align_corners=False)
            cam = cam[0, 0].cpu().numpy()

            # normalize EACH layer's CAM independently before combining --
            # different layers' raw activation/gradient magnitudes aren't
            # on comparable scales, so averaging un-normalized CAMs would
            # let whichever layer happens to have larger raw values
            # dominate for reasons that have nothing to do with which
            # layer is actually more informative.
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            per_layer_cams.append(cam)

        cam = np.mean(per_layer_cams, axis=0)
        # re-normalize the combined map too -- averaging several already-
        # [0,1] maps doesn't necessarily still span the full [0,1] range
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


def visualize_clip(model, data, raw_frames, target_layers=None, out_path=None, fps=4, alpha=0.45):
    """
    Same Grad-CAM computation as visualize_frame, but overlays EVERY frame
    in the clip (not just one) and saves the sequence as an animated GIF.

    raw_frames: (T, H, W, 3) or (T, H, W) -- same requirement as
                visualize_frame: these must be the exact frames (same
                indices, same resize) that went into `data["clip"]`.
    target_layers: module, or list of modules, to hook. Defaults to
                [model.model.layer3, model.model.layer4] -- see
                GradCAM3D's docstring for why fusing a frozen, temporally-
                richer layer with the only actually-trained layer is the
                point here, rather than using layer4 alone. Pass a single
                layer (e.g. just model.model.layer4) to get the original,
                single-layer behavior back.
    fps: playback speed of the output GIF.
    """
    if target_layers is None:
        target_layers = [model.model.layer3, model.model.layer4]
    gradcam = GradCAM3D(model, target_layers)
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


def load_clip_from_row(row, config, video_root="."):
    """
    Read one video straight from a dataframe row (no Dataset/DataLoader
    involved), sample a single clip's worth of frames, and produce both:
      - raw_frames: (T, H, W, 3) uint8 RGB, for a human-viewable overlay
      - clip_tensor: (1, C, T, H, W), the exact tensor the model expects

    This mirrors what VideoDataset does internally, just done directly and
    deterministically (center clip, no augmentation) so the frames you
    overlay on are guaranteed to be the same ones the model actually saw.
    """
    from datasets import img2tensor, imgResize, video_loader

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

    return raw_frames, clip_tensor


def run_gradcam(fold_idx, row_selector=0, fps=4, video_root=".",
                 checkpoint_name="best_combined_model.pth", out_dir=".",
                 target_layers="layer3,layer4"):
    """
    fold_idx: which fold's checkpoint + validation video to use (1-5)
    row_selector: either an int (positional row index within that fold's
                  validation rows) or a string (a specific Study ID to look
                  up within that fold)
    fps: playback speed of the output GIF
    target_layers: comma-separated resnet stage name(s) to hook for
                  Grad-CAM, e.g. "layer3,layer4" (default -- fuses a
                  temporally-richer frozen layer with the only actually-
                  trained layer, see GradCAM3D's docstring) or just
                  "layer4" for the original single-layer behavior.
    """
    import sys
    sys.path.append(".")
    from config import GradCamConfig
    from train_cv import get_model
    import pandas as pd

    config = GradCamConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load the checkpoint that belongs to THIS fold ---
    checkpoint_path = os.path.join(config.checkpoint_dir, f"fold-{fold_idx}", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- check config.checkpoint_dir "
            f"and that fold {fold_idx} finished training."
        )
    model = get_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    target_layer_names = [n.strip() for n in target_layers.split(",")]
    resolved_layers = [getattr(model.model, name) for name in target_layer_names]
    print(f"Grad-CAM target layer(s): {target_layer_names}")

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
            raise ValueError(f"Study ID '{row_selector}' not found in fold {fold_idx}'s validation set")
        row = matches.iloc[0]
    else:
        row = val_df.iloc[row_selector]

    print(f"Fold {fold_idx} | Subject: {row['Study ID']} | Side: {row.get('Side','?')} | "
          f"true label: {row['label']} | video: {row['Video']}")

    # --- read the video, build raw frames + model input tensor together ---
    raw_frames, clip_tensor = load_clip_from_row(row, config, video_root=video_root)
    single = {"clip": clip_tensor.to(device)}

    out_path = os.path.join(out_dir, f"gradcam_fold{fold_idx}_{row['Study ID']}.gif")
    overlays, cam, prob = visualize_clip(model, single, raw_frames, target_layers=resolved_layers,
                                          out_path=out_path, fps=fps)
    print(f"Predicted prob: {prob:.3f}  |  true label: {row['label']}")
    return overlays, cam, prob


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM GIF for a specific fold's checkpoint + video")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (1-5)")
    parser.add_argument("--row", default="0",
                         help="Row selector within that fold's val set: an integer position "
                              "(e.g. '0') or a Study ID string (e.g. 'SUBJ01')")
    parser.add_argument("--fps", type=int, default=4, help="Playback speed of the output GIF")
    parser.add_argument("--video_root", default=".", help="Root directory the CSV's Video paths are relative to")
    parser.add_argument("--checkpoint_name", default="best_combined_model.pth")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--target_layers", default="layer3,layer4",
                         help="Comma-separated resnet stage name(s) to hook for Grad-CAM. "
                              "Default 'layer3,layer4' fuses a temporally-richer frozen layer "
                              "with the only actually-trained layer -- pass just 'layer4' for "
                              "the original single-layer behavior.")
    args = parser.parse_args()

    row_selector = int(args.row) if args.row.isdigit() else args.row

    run_gradcam(
        fold_idx=args.fold,
        row_selector=row_selector,
        fps=args.fps,
        video_root=args.video_root,
        checkpoint_name=args.checkpoint_name,
        out_dir=args.out_dir,
        target_layers=args.target_layers,
    )