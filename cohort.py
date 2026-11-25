"""Validate the subject-disjoint Healthy/Latent research cohort."""
import pandas as pd


def healthy_latent_cohort(data, n_folds=5):
    df = data.copy()
    required = ['Study ID', 'MTrP type', 'label', 'cv_fold', 'Video', 'Side']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing cohort columns: {sorted(missing)}')
    df = df[df['MTrP type'].isin(['Healthy', 'Latent'])].copy()
    if df.empty:
        raise ValueError('No Healthy/Latent rows')
    expected = df['MTrP type'].map({'Healthy': 0, 'Latent': 1})
    if not pd.to_numeric(df.label, errors='coerce').eq(expected).all():
        raise ValueError('Healthy must be label 0; Latent must be label 1')
    folds = pd.to_numeric(df.cv_fold, errors='coerce')
    if not folds.isin(range(1, n_folds + 1)).all():
        raise ValueError('Every Healthy/Latent row needs a valid integer cv_fold')
    df['cv_fold'] = folds.astype(int)
    df['label'] = expected.astype(int)
    if df[required].isna().any().any():
        raise ValueError('Missing cohort metadata')
    if df.groupby('Study ID').cv_fold.nunique().gt(1).any():
        raise ValueError('Subject appears in multiple folds')
    if df.groupby(['Study ID', 'Side']).label.nunique().gt(1).any():
        raise ValueError('Inconsistent labels within a subject/side')
    if df.Video.duplicated().any():
        raise ValueError('Duplicate video rows')
    for fold in range(1, n_folds + 1):
        for part in (df[df.cv_fold == fold], df[df.cv_fold != fold]):
            if part.label.nunique() != 2:
                raise ValueError(f'Fold {fold} train/test must each contain both classes')
    return df.reset_index(drop=True)
