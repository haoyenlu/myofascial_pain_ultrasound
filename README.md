# Ultrasound MPS Video Classification

Research code for classifying musculoskeletal ultrasound **videos** (latent
myofascial trigger point vs. healthy muscle) with subject-level
cross-validation.

The main model is a **Video Diffusion Encoder (VDE)**. A 2D denoising-diffusion
U-Net is pretrained on ultrasound frames, separately for each CV fold. It is
then frozen and used as a feature extractor: frames are noised to timestep
`t`, intermediate U-Net features from one decoder/encoder block are pooled,
and a small 3D-convolutional head classifies the resulting feature clip.
Optional demographic and clinical inputs (sex, age, pre/post pressure-pain
threshold) can be fused into the head.

Baselines in this repository: a Kinetics-pretrained 3D ResNet-50, SimCLR
pretraining followed by fine-tuning, VideoMAE pretraining followed by
fine-tuning, and demographics-only and PPT-only models.

> **Data and weights are not included.** This repository contains source
> code only: no patient videos, clinical metadata, or weights trained on
> patient data. See [`data/README.md`](data/README.md) for the input format.

## Installation

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
# install the torch/torchvision build for your CUDA version first
pip install -r requirements.txt
```

For the 3D ResNet / SimCLR baselines, download the Kinetics-pretrained
`resnet-50-kinetics.pth` from
[video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
into `pretrained/`.

Run all commands from the repository root.

## Pipeline

### 1. Prepare metadata and folds

Create one or more metadata CSVs following [`data/README.md`](data/README.md),
then assign subject-level, stratified folds:

```bash
python -m utils.build_folds my_metadata.csv --out data/all_data_combined.csv
```

`StratifiedGroupKFold` assigns each subject's videos, both sides and both
time points, to a single fold, so no subject's data can leak from training
into validation.

### 2. Pretrain the per-fold diffusion prior

```bash
python train_diffusion.py            # writes checkpoints/<DiffusionConfig.checkpoint_dir>/fold-k/
```

The prior for fold *k* is trained only on fold *k*'s training subjects.
Afterwards, point `VDEConfig.diffusion_fold_checkpoint` in `config.py` at the
milestone you want to use.

### 3. Train and evaluate the VDE classifier (5-fold CV)

```bash
python train_cv.py --folds all --timestep 50 --block_to_use 13
python train_cv.py --folds all --use_demo            # fuse sex/age/PPT
python train_cv.py --infer --folds all               # re-run inference from saved checkpoints
python train_cv.py --help                            # all options
```

Each fold writes `training_history.csv`, `oof_val_result.csv` and its
checkpoint. Out-of-fold predictions are then pooled across all folds
(`pooled_oof.csv`). Clip probabilities are averaged to give video-level
predictions.

To sweep timesteps and U-Net blocks:

```bash
VDE_TIMESTEPS="0 50 250" VDE_BLOCKS="5 13 17" bash run_vde_ablation.sh
```

### 4. Explainability

```bash
python vde_gradcam.py --help         # Grad-CAM for VDE
python -m utils.gradcam3d --help     # Grad-CAM for the 3D ResNet
python -m videomae.gradcam --help
python -m simclr.gradcam --help
```

### Baselines

```bash
python train_cv.py --model resnet50 --folds all        # Kinetics 3D ResNet-50
python -m simclr.pretrain --help && python -m simclr.finetune --help
python -m videomae.pretrain --help && python -m videomae.finetune --help
python -m baselines.train_demo_only_cv --help         # logistic regression on demographics
python -m baselines.train_ppt_embedding_cv --help     # PPT-only
python evaluate_diffusion_linear_probe.py --help      # linear probe on frozen diffusion features
```

## Preprocessing and augmentation

- **Loading** (`datasets.py`): videos are decoded with OpenCV, converted to
  grayscale, resized to 224x224 (bicubic), and scaled to `[0, 1]`
  (`[-1, 1]` for diffusion pretraining).
- **Clip sampling** (`vidaug/augmentors/sampler.py`): sliding-window
  sampling with `clip_len` frames and stride `step_size`. Videos shorter
  than `clip_len` are padded by ping-pong looping.
- **Augmentation** (`vidaug/`): per-clip-consistent rotation, translation,
  shear, Gaussian blur, and flips, applied with probability
  `augmentation_prob`, to training clips only.
- **Normalization of clinical covariates**: the age and PPT scalers are fit
  on each fold's training split only.

## Repository layout

```text
config.py                 all experiment settings (paths, hyperparameters)
datasets.py               video loading and frame preprocessing
cohort.py                 metadata validation (leakage/label checks)
reproducibility.py        seeding
train_diffusion.py        per-fold diffusion-prior pretraining
train_cv.py               VDE / 3D ResNet cross-validation training and inference
vde_gradcam.py            VDE Grad-CAM
evaluate_diffusion_linear_probe.py
run_vde_ablation.sh       timestep/block sweep
models/                   VDE, 3D ResNet, diffusion U-Net
vidaug/                   video augmentation and clip samplers
simclr/, videomae/        self-supervised baselines
baselines/                demographics-only and PPT-only models
utils/                    fold building, frame counting, ROC plots, 3D Grad-CAM
```

## Notes on evaluation

- Checkpoints are selected on the outer validation fold, using an
  EMA-smoothed `val_auc - val_loss`. The pooled OOF metrics therefore also
  served as the model-selection criterion and are not an independent test
  estimate.
- Youden-J thresholds are fit on training-split predictions.

## License

Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Third-party components are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

This software is for research only. It is not a medical device and must not
be used for clinical diagnosis.
