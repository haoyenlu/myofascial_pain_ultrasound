"""Image-only, fixed-hyperparameter probe of each fold's frozen diffusion prior.

Uses Pre videos only and averages videos within subject/side before fitting.
No outer-fold labels are used for feature scaling, fitting or model selection.
Existing priors without training manifests yield exploratory results only.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from config import Config, VDEConfig
from datasets import video_loader, imgResize
from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet
from cohort import healthy_latent_cohort


def pooled_features(model, video, frames, block, timestep, seed, device):
    images = video_loader(video)
    if not images:
        raise ValueError(f'Empty video: {video}')
    indices = np.linspace(0, len(images) - 1, min(frames, len(images)), dtype=int)
    x = torch.from_numpy(np.stack([np.array(imgResize(images[i], 224)) for i in indices])).float().unsqueeze(1).to(device) / 255
    with torch.inference_mode():
        feature = model.get_unet_feature_at(x, t=timestep, block_idx=block,
                    deterministic=True, seed=seed, batch_invariant=True)
        # Pool spatially and temporally; no learned supervised projection.
        return feature.mean(dim=(0, 2, 3)).cpu().numpy()


def fit_probe(train_x, test_x, labels, weights):
    scaler = StandardScaler().fit(train_x, sample_weight=weights)
    classifier = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
    classifier.fit(scaler.transform(train_x), labels, sample_weight=weights)
    return classifier.predict_proba(scaler.transform(test_x))[:, 1]


def training_weights(units):
    weights = 1 / units.groupby('Study ID')['Study ID'].transform('size').to_numpy()
    y = units.label.to_numpy()
    for label in (0, 1):
        weights[y == label] /= weights[y == label].sum()
    return weights * (len(weights) / weights.sum())


def random_control(df, cfg, args, out, device):
    """Same feature architecture and probe, without self-supervised training."""
    torch.manual_seed(42)
    model = GaussianDiffusion(Unet(**cfg.unet_param), **cfg.diffusion_param).to(device)
    model.eval().requires_grad_(False)
    x = np.stack([pooled_features(model, row.Video, args.frames, args.block, args.timestep, args.seed, device) for row in df.itertuples()])
    cols = [f'f{i}' for i in range(x.shape[1])]
    table = pd.concat([df.reset_index(drop=True), pd.DataFrame(x, columns=cols)], axis=1)
    units = table.groupby(['Study ID', 'Side', 'cv_fold', 'label', 'source_file'], as_index=False)[cols].mean()
    results = []
    for fold in range(1, 6):
        train = units.cv_fold != fold
        weights = training_weights(units.loc[train])
        result = units.loc[~train, ['Study ID', 'Side', 'cv_fold', 'label']].copy()
        result['probability'] = fit_probe(units.loc[train, cols], units.loc[~train, cols], units.loc[train, 'label'], weights)
        results.append(result)
    result = pd.concat(results, ignore_index=True)
    result.to_csv(out / 'random_backbone_oof.csv', index=False)
    return float(roc_auc_score(result.label, result.probability))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', default='checkpoints/2d_diffusion_cv')
    parser.add_argument('--output', default='results/diffusion_hl_linear_probe')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--block', type=int, default=17)
    parser.add_argument('--timestep', type=int, default=50)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--random-control-only', action='store_true', help='Add an untrained-backbone control to an existing completed run')
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('frames must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = healthy_latent_cohort(pd.read_csv(Config.data_csv))
    df = df[df.Intervention == 'Pre'].reset_index(drop=True)
    df.to_csv(out / 'evaluation_cohort.csv', index=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    provenance = []
    cfg = VDEConfig()
    if args.random_control_only:
        metrics_path = out / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        for key in ('frames', 'block', 'timestep', 'seed'):
            if metrics['settings'][key] != getattr(args, key):
                raise ValueError(f'Control settings differ: {key}')
        metrics['random_backbone_auc'] = random_control(df, cfg, args, out, device)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
        return
    for fold in range(1, 6):
        fold_dir = Path(args.checkpoint_root) / f'fold-{fold}'
        manifest = fold_dir / 'training_manifest.csv'
        if manifest.exists():
            trained = pd.read_csv(manifest)
            assert not set(trained['Study ID']) & set(df.loc[df.cv_fold == fold, 'Study ID']), 'Pretraining leakage'
            assert set(df.loc[df.cv_fold != fold, 'Study ID']) == set(trained['Study ID']), 'Pretraining cohort mismatch'
        provenance.append(dict(fold=fold, manifest_verified=manifest.exists()))
        model = GaussianDiffusion(Unet(**cfg.unet_param), **cfg.diffusion_param).to(device)
        checkpoint = fold_dir / 'model-20.pt'
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval().requires_grad_(False)
        embeddings = []
        for i, row in df.iterrows():
            embeddings.append(pooled_features(model, row.Video, args.frames, args.block, args.timestep, args.seed, device))
            if (i + 1) % 30 == 0:
                print(f'Fold {fold}: extracted {i+1}/{len(df)} videos', flush=True)
        x = np.stack(embeddings)
        np.savez_compressed(out / f'fold-{fold}_features.npz', features=x)
        feature_cols = [f'f{i}' for i in range(x.shape[1])]
        table = pd.concat([df.reset_index(drop=True), pd.DataFrame(x, columns=feature_cols)], axis=1)
        units = table.groupby(['Study ID', 'Side', 'cv_fold', 'label', 'source_file'], as_index=False)[feature_cols].mean()
        train = units.cv_fold != fold
        # Equal subject contribution, then equal total weight per class.
        train_units = units.loc[train]
        weights = training_weights(train_units)
        y = train_units.label.to_numpy()
        probabilities = fit_probe(units.loc[train, feature_cols], units.loc[~train, feature_cols], y, weights)
        result = units.loc[~train, ['Study ID', 'Side', 'cv_fold', 'label', 'source_file']].copy()
        result['probability'] = probabilities
        # Acquisition-source-only comparator; no patient measurements.
        sources = pd.get_dummies(units.source_file, dtype=float)
        result['source_only_probability'] = fit_probe(sources.loc[train], sources.loc[~train], y, weights)
        result.to_csv(out / f'fold-{fold}_predictions.csv', index=False)
        all_results.append(result)
        print(f'Fold {fold} AUC: {roc_auc_score(result.label, probabilities):.4f}', flush=True)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pooled = pd.concat(all_results, ignore_index=True)
    pooled.to_csv(out / 'pooled_oof_subject_side.csv', index=False)
    metrics = dict(roc_auc=roc_auc_score(pooled.label, pooled.probability),
                   balanced_accuracy_at_0_5=balanced_accuracy_score(pooled.label, pooled.probability >= 0.5),
                   source_only_auc=roc_auc_score(pooled.label, pooled.source_only_probability),
                   n_subjects=int(pooled['Study ID'].nunique()), n_subject_sides=len(pooled),
                   settings=vars(args), pretraining_provenance=provenance,
                   interpretation='Exploratory: historical splits/block were previously inspected; priors lacking manifests have unverified training provenance. No hyperparameters selected in this run.')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
