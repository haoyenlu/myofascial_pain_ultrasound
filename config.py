"""Experiment configuration.

All paths are relative to the repository root. Patient data is NOT included in
this repository; see data/README.md for the expected metadata CSV schema.
"""

class Config:
    seed = 42  # Fold i uses seed + i - 1; evaluation noise has its own eval_seed.
    # Single combined, re-folded dataset -- see build_folds.py.
    # No separate holdout: with ~38 subjects, a fixed 5-subject test set
    # was too small to trust (we measured subject-level AUC = 0.50 on it).
    # All subjects now go through 5-fold CV; the pooled out-of-fold (OOF)
    # predictions across all 5 folds are the primary evaluation metric.
    data_csv = "data/all_data_combined.csv"

    n_fold = 5
    runs = 1

    model = "VDE"
    repeat_chn = False

    epochs = 80
    warmup_epochs = 10
    batch_size = 32
    lr = 3e-3
    min_lr = 1e-5
    weight_decay = 3e-4        # lowered from 1e-2 -- see discussion on calibration
    no_decay_norm_and_bias = True  # exclude 1D params (biases, norm gains) from weight decay
    grad_clip_norm = 5.0       # clips the rare spike (we saw grad norms up to 15.8 on
                                # trailing partial batches); set high enough that normal
                                # steps (mostly 1.5-6) pass through untouched
    ema_decay = 0.9
    ema_loss_alpha = 0.8       # Current epoch weight for both EMA loss and AUC.

    # Checkpoint selection: use the EMA-smoothed combined score, not raw
    # per-epoch AUC or loss. A single epoch's AUC can spike from validation
    # noise (we saw fold-level swings of 0.03 to 0.93 within one run) --
    # smoothing requires a *sustained* good stretch, not a lucky epoch.
    checkpoint_metric = "combined_score"  # = ema_val_auc - ema_val_loss

    result_dir = "./results/resnet_cv_v2"
    checkpoint_dir = "./checkpoints/resnet_cv_v2"

    train_folds = [1,2,3,4,5]

    step_size = 6
    clip_len = 12
    frame_size = 224

    augmentation = True
    augmentation_prob = 0.75
    rotation_degree = 20
    gaussian_blur_sigma = 1.0
    translation_degree = 10
    shear_degree = 10

    use_demo = False


class ResnetConfig(Config):
    resnet_param = {
        "sample_size": Config.frame_size,
        "sample_duration": Config.clip_len,
        "shortcut_type": "B",
        "num_classes": 1,
        "last_fc": True,
    }
    resnet_pretrained_path = "./pretrained/resnet-50-kinetics.pth"

    # These are the layers we intend to freeze (kept trainable = the head).
    # Fixed bug: previous code set `param.require_grad` (typo, silently a
    # no-op) instead of `param.requires_grad`, so the "frozen" backbone was
    # actually being fully fine-tuned the whole time.
    freeze_layers = ("conv1.", "layer1.", "layer2.", "layer3.")


class GradCamConfig(ResnetConfig):
    clip_len = 12
    checkpoint_dir = "./checkpoints/resnet50_cv_12f"


class SimCLRFinetuneConfig(ResnetConfig):
    """
    Config for finetuning the classifier starting from a SimCLR-pretrained
    encoder (simclr/pretrain.py -> simclr/finetune.py), reusing train_cv.py's
    training pipeline UNCHANGED -- see simclr/finetune.py's module docstring
    for how it does that (monkey-patching get_model so the backbone's
    starting weights come from the SimCLR checkpoint instead of raw
    Kinetics, while everything else -- freezing, LR schedule, checkpoint
    selection, pooled OOF eval -- stays exactly train_cv.py's own logic).

    Inherits from ResnetConfig, not SimCLRConfig: this describes the
    CLASSIFICATION finetuning stage, which train_cv.py's pipeline expects
    to be a ResnetConfig-shaped config (resnet_param, freeze_layers, LR
    schedule, etc.). SimCLRConfig is a deliberately separate, standalone
    config for the earlier, different SimCLR PREtraining stage (contrastive
    loss, augmentation views, temperature) -- see its own docstring for why
    the two are kept apart.

    Only checkpoint_dir is overridden here, to its own directory -- distinct
    from both:
      - Config.checkpoint_dir ("./checkpoints/simclr_cv_12f"), which the
        plain Kinetics-init ResnetConfig baseline also uses. Without this
        override, running simclr/finetune.py would write into that SAME
        directory and silently overwrite the baseline's (or a different
        SimCLR run's) checkpoints/training_history/OOF results.
      - SimCLRConfig.checkpoint_dir ("./checkpoints/simclr"), which holds
        the upstream PREtraining checkpoints (simclr_encoder.pt etc.) this
        config's run loads FROM -- not where its own classification
        checkpoints should be written TO.

    Every other hyperparameter (lr, epochs, batch_size, freeze_layers, ...)
    is inherited unchanged from ResnetConfig for now. If you want the
    SimCLR-finetuned run to use different hyperparameters than the plain
    baseline (e.g. a lower LR, since the encoder starts already
    domain-adapted rather than Kinetics-only), override them here.
    """
    checkpoint_dir = "./checkpoints/simclr_finetune_cv_12f"





class SimCLRConfig:
    """
    Standalone config for both SimCLR self-supervised pretraining
    (pretrain_simclr.py) and finetuning from a SimCLR checkpoint
    (finetune_simclr.py). Deliberately does NOT inherit from Config --
    every field it needs is spelled out here, so changing Config for a
    classification experiment (e.g. a clip_len sweep) can never silently
    change what SimCLR pretraining does, and vice versa.
    """
    # --- data / sampling ---
    # Uses the SAME va.SlidingWindowSampler(clip_len, step) as
    # train_cv.py's VideoDataset -- keeps the pretraining data distribution
    # identical to what finetuning will see (same clips, same per-video
    # window counts, same weighting of long vs. short videos).
    data_csv = "data/all_data_combined.csv"
    clip_len = 12
    step_size = 6
    frame_size = 224
    repeat_chn = True

    # --- augmentation (used to build the two SimCLR views) ---
    rotation_degree = 20
    translation_degree = 10
    shear_degree = 10
    gaussian_blur_sigma = 1.0

    # --- SimCLR pretraining hyperparameters ---
    epochs = 100
    lr = 3e-4
    batch_size = 16
    weight_decay = 1e-6
    temperature = 0.1
    proj_dim = 128
    grad_clip_norm = 5.0

    # --- checkpoints ---
    checkpoint_dir = "./checkpoints/simclr"
    resnet_pretrained_path = "./pretrained/resnet-50-kinetics.pth"

    # --- backbone architecture (must match the classifier's, since the
    # encoder gets loaded into the same resnet50(**resnet_param) later) ---
    resnet_param = {
        "sample_size": frame_size,
        "sample_duration": clip_len,
        "shortcut_type": "B",
        "num_classes": 1,
        "last_fc": True,
    }

    # --- freezing, for FINETUNING only (see note below) ---
    # NOT used during SimCLR pretraining -- build_encoder() in
    # pretrain_simclr.py leaves every layer trainable, since the whole
    # point of pretraining is to adapt ALL layers (especially the early,
    # most Kinetics-specific ones) toward ultrasound. This is only read by
    # finetune_simclr.py, which DOES freeze layer1-3 during classification
    # finetuning, per the earlier discussion. Don't add a freeze step to
    # pretrain_simclr.py based on this attribute's presence.
    freeze_layers = ("conv1.", "layer1.", "layer2.", "layer3.")

    # If True: freeze the ENTIRE backbone (conv1 through layer4) and only
    # train the final fc layer -- i.e. linear probing, the standard
    # self-supervised-representation evaluation protocol. Much lower
    # overfitting risk given ~38 subjects, but removes any chance for the
    # model to correct misalignment between SimCLR's contrastive objective
    # and the actual classification boundary. Treat this as a comparison
    # arm against the default (layer4 trainable), not an assumed winner --
    # linear_probe_auc() in evaluate_simclr_quality.py measures roughly this
    # same thing on frozen embeddings, so check that first if you want a
    # cheap preview before running a full finetuning job either way.
    linear_probe_only = False


class VMAEConfig:
    """
    Standalone config for VideoMAE (ViT-based masked autoencoder) fold-
    restricted pretraining (pretrain_videomae.py) and finetuning
    (finetune_videomae.py). Named VMAEConfig, not VideoMAEConfig, to avoid
    colliding with `transformers.VideoMAEConfig` (the HF library's own
    config class for the model architecture itself -- that one holds
    hidden_size/num_hidden_layers/etc.; this one holds OUR experiment
    settings: which checkpoint to start from, fold restriction, optimizer
    hyperparameters, freezing strategy).

    Deliberately standalone (no inheritance from Config or SimCLRConfig) --
    same reasoning as SimCLRConfig: this pretraining pipeline's settings
    should never silently change because a classification experiment
    changed Config.

    Architecture note: num_frames/patch_size/tubelet_size/image_size are
    NOT set here -- they come from the pretrained checkpoint's own config
    (loaded via VideoMAEForPreTraining.from_pretrained(pretrained_model_name)
    at runtime) and are read off model.config directly. Overriding them
    would require care around the sinusoidal position embeddings and isn't
    attempted here; num_frames=16 (the checkpoint's default) is what this
    pipeline uses for a fold's sampled clip length, decoupled from
    Config.clip_len / SimCLRConfig.clip_len, same as those two are already
    decoupled from each other.
    """
    # --- data / sampling ---
    data_csv = "data/all_data_combined.csv"
    step_size = 6       # sliding-window step for the fold-restricted clip sampler
                         # (num_frames itself comes from the checkpoint's config, see above)

    # --- checkpoint sources ---
    # The RAW MAE-pretrained-on-Kinetics-400 checkpoint (encoder + decoder,
    # no classification head) -- NOT "MCG-NJU/videomae-base-finetuned-kinetics",
    # which is already finetuned for Kinetics-400 action classification and
    # is the wrong starting point for continuing self-supervised pretraining.
    pretrained_model_name = "MCG-NJU/videomae-base"
    checkpoint_dir = "./checkpoints/videomae"

    # --- pretraining hyperparameters ---
    mask_ratio = 0.9    # paper default -- VideoMAE's key finding is that video
                         # redundancy across frames means a very high masking
                         # ratio (much higher than image MAE) still works
    epochs = 50          # fewer than SimCLR's 100 -- this is continued
                         # pretraining of an already-strong Kinetics checkpoint
                         # on a small unlabeled corpus (~800-1000 clips across
                         # a fold's training videos), not pretraining from zero
    lr = 1e-4
    batch_size = 8        # ViT-B over 1568-token sequences is memory-heavier
                           # than the ResNet50/SimCLR pipeline -- start smaller
                           # and increase if your GPU has headroom
    weight_decay = 0.05    # typical ViT/MAE value, much higher than SimCLR's
                            # 1e-6 -- transformers overfit differently than CNNs
    warmup_epochs = 5
    grad_clip_norm = 5.0
    num_frames = 12
    image_size = 224

    # --- finetuning: freezing strategy ---
    # ViT-Base has 12 transformer blocks (config.num_hidden_layers), no
    # resnet-style named layers to freeze by string match. Analogous to the
    # ResNet pipeline's "freeze early layers, adapt the last block" pattern:
    # freeze the patch embeddings (always) and the first `num_frozen_blocks`
    # transformer blocks; leave the rest + classifier head trainable.
    num_frozen_blocks = 8   # out of 12 -- leaves the last 4 blocks trainable

    # If True: freeze the ENTIRE encoder (embeddings + all 12 blocks), only
    # the classification head (fc_norm + classifier) trains -- linear
    # probing, same idea and same caveats as SimCLRConfig.linear_probe_only.
    linear_probe_only = False

    # --- finetuning: classification hyperparameters ---
    ft_epochs = 50
    ft_lr = 1e-4
    ft_batch_size = 16
    ft_weight_decay = 1e-4
    ft_warmup_epochs = 10
    ema_loss_alpha = 0.2


class DiffusionConfig:
    """
    Config for the 2D diffusion-prior pretraining step.

    Previously this pointed at 5 separate per-fold CSVs (data/cv_fold_1.csv
    ... cv_fold_5.csv) plus a numeric-PID frame-directory scheme
    (cropped_frames/<subject>/...) that doesn't correspond to the
    actual Study ID / Video-path columns used by the classification
    pipeline. That's been replaced: this now reads the single combined,
    re-folded CSV (see build_folds.py) and splits by the `cv_fold` column,
    so the diffusion prior and the classifier are trained on the exact
    same fold assignment -- important since a mismatched split would let
    the diffusion model see videos that ended up in a classifier fold's
    validation set.
    """

    data_csv = "data/all_data_combined.csv"
    seed = 42
    n_fold = 5

    frame_size = 224
    batch_size = 8

    checkpoint_dir = "./checkpoints/2d_diffusion_hl_v2"
    max_frames_per_video = 32
    # Set to an int (e.g. 20) to resume from that milestone; None = fresh start.
    milestone = None

    unet_param = {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "channels": 1,
        "attn_heads": 4,
    }

    diffusion_param = {
        "timesteps": 1000,
        "beta_schedule": "cosine",
    }

    aug_param = {
        "random_horizontal_flip_prob": 0.5,
        "random_vertical_flip_prob": 0.5,
        "random_gaussian_blur_kernel": 3,
        "random_gaussian_blur_sigma": 1.0,
        "random_gaussian_blur_prob": 0.5,
        "random_affine_degree": 20,
        "random_affine_translate": (0.1, 0.1),
        "random_affine_scale": (0.9, 1.1),
        "random_affine_shear": 10,
    }

    trainer_param = {
        "calculate_fid": False,
        "gradient_accumulate_every": 4,
        "train_lr": 1e-4,
        "train_num_steps": 10000,
        "save_and_sample_every": 500,
        "num_samples": 16,
        "num_fid_samples": 16,
    }


class VDEConfig(Config):
    """Video Diffusion Encoder + 3D ResNet head, with optional demographic
    (sex/age/PPT) fusion. Plugs into the same 5-fold CV pipeline as the
    resnet configs -- see train_cv.py's `get_vde_model`.
    """
    model = "VDE"
    use_demo = False
 
    # The diffusion U-Net is always frozen by VDE/DiffusionEncoder.
    # This controls additional freezing of the projection/classifier only.
    freeze_layers = ()
 
    """Diffusion Model Parameters"""
    timestep = 50
 
    diffusion_fold_checkpoint = {
        "fold-1": "checkpoints/2d_diffusion_cv/fold-1/model-20.pt",
        "fold-2": "checkpoints/2d_diffusion_cv/fold-2/model-20.pt",
        "fold-3": "checkpoints/2d_diffusion_cv/fold-3/model-20.pt",
        "fold-4": "checkpoints/2d_diffusion_cv/fold-4/model-20.pt",
        "fold-5": "checkpoints/2d_diffusion_cv/fold-5/model-20.pt",
    }
 
    # Backbone (U-Net) used by the diffusion model
    unet_param = {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "channels": 1,
        "attn_heads": 4,
    }
 
    diffusion_param = {
        "image_size": 224,
        "timesteps": 1000,
        "beta_schedule": "cosine",
    }
 
    # Extracts per-frame diffusion features fed into the 3D ResNet head
    encoder_param = {
        "image_size": 224,
        "block_to_use": 13,
        "pool_size": 32,  # Use --encoder_pool_size 64 for the larger variant.
        "timestep": timestep,
        "num_noise_samples": 1,
        "eval_seed": 12,
        "batch_invariant_eval": True,
    }
 
    resnet_head_param = {
        "input_dim": encoder_param["pool_size"],  # Spatial input size, not channel count.
        "planes_3d_blocks": [16, 32],
        "pool_size": 2,
        "hidden_dim": 64,
        "dropout_prob": 0.3,
        "stride": 2,
        "block_to_use": 13,
    }
 
    # Demographic embedding dims, consumed via get_vde_model's
    # sex/age/ppt_embedding_dim + num_ppt args
    demo_param = {
        "sex_embedding": 4,
        "age_embedding": 4,
        "ppt_embedding": 4,  # Per time point: separate pre/post embeddings (8 total).
        "num_ppt": 2,
        "n_classes": 1,
    }
 
    checkpoint_dir = "./checkpoints/vde_cv_t50_1noise_sample"
