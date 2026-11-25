"""Logistic regression on age, sex, and separate pre/post PPT; existing CV folds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from cohort import healthy_latent_cohort
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--penalty', choices=['l1', 'l2'], default='l2')
    parser.add_argument('--cv', choices=['fivefold', 'loso'], default='fivefold',
                        help='Leave one subject out keeps both sides together.')
    parser.add_argument('--ppt-only', action='store_true', help='Use only pre/post PPT; exclude age and sex.')
    parser.add_argument('--average-ppt', action='store_true',
                        help='Replace separate pre/post PPT inputs with their arithmetic mean.')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = healthy_latent_cohort(pd.read_csv(Config.data_csv))
    continuous = ['PPT_PRE_AVG', 'PPT_POST_AVG'] if args.ppt_only else ['Age', 'PPT_PRE_AVG', 'PPT_POST_AVG']
    clinical = continuous + ([] if args.ppt_only else ['Sex'])
    keys = ['Study ID', 'Side']
    if source.groupby(keys)[clinical + ['label']].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting clinical values within subject/side')
    df = source.drop_duplicates(keys).copy()
    if df[clinical].isna().any().any() or (not args.ppt_only and not df.Sex.isin(['M', 'F']).all()):
        raise ValueError('Missing or invalid clinical inputs')
    if not np.isfinite(df[continuous].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite clinical values')
    if args.average_ppt:
        df['PPT_MEAN'] = (df['PPT_PRE_AVG'] + df['PPT_POST_AVG']) / 2
        continuous = [c for c in continuous if c not in ('PPT_PRE_AVG', 'PPT_POST_AVG')] + ['PPT_MEAN']
    if not args.ppt_only:
        df['sex_code'] = df.Sex.map({'M': 0, 'F': 1})
    features = continuous + ([] if args.ppt_only else ['sex_code'])
    outputs, metrics, coefficients = [], [], []
    splits = ([(i, df[df['Study ID'] != subject], df[df['Study ID'] == subject])
               for i, subject in enumerate(sorted(df['Study ID'].unique()), 1)]
              if args.cv == 'loso' else
              [(fold, df[df.cv_fold != fold], df[df.cv_fold == fold]) for fold in range(1, 6)])
    for fold, train, val in splits:
        assert not set(train['Study ID']) & set(val['Study ID'])
        model = Pipeline([
            ('scale', ColumnTransformer([('continuous', StandardScaler(), continuous)], remainder='passthrough')),
            ('classifier', LogisticRegression(C=1.0, penalty=args.penalty, class_weight='balanced',
                                               solver='liblinear' if args.penalty == 'l1' else 'lbfgs',
                                               max_iter=2000, random_state=args.seed)),
        ])
        model.fit(train[features], train.label)
        train_probs = model.predict_proba(train[features])[:, 1]
        fpr, tpr, thresholds = roc_curve(train.label, train_probs, drop_intermediate=False)
        valid = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= train_probs.max()))
        threshold = float(thresholds[valid[np.argmax((tpr-fpr)[valid])]])
        probability = model.predict_proba(val[features])[:, 1]
        result = val[['Study ID', 'Side', 'label', 'cv_fold']].rename(
            columns={'Study ID': 'subject', 'Side': 'side', 'cv_fold': 'fold'}).copy()
        if args.cv == 'loso':
            result = result.rename(columns={'fold': 'original_cv_fold'})
            result['fold'] = fold
        result['probability'] = probability
        result['youden_threshold'] = threshold
        result['youden_prediction'] = (probability >= threshold).astype(int)
        directory = out / f'fold-{fold}'
        directory.mkdir(exist_ok=True)
        joblib.dump({'pipeline': model, 'features': features, 'youden_threshold': threshold},
                    directory / 'logistic_model.joblib')
        result.to_csv(directory / 'oof_val_result.csv', index=False)
        outputs.append(result)
        metrics.append(dict(fold=fold, n=len(val),
                            held_out_subjects=','.join(map(str, val['Study ID'].unique())),
                            auc=roc_auc_score(val.label, probability) if val.label.nunique() == 2 else None,
                            youden_threshold=threshold))
        clf = model.named_steps['classifier']
        coefficients.append(dict(fold=fold, intercept=float(clf.intercept_[0]),
                                 **dict(zip(features, clf.coef_[0].tolist()))))
        print(metrics[-1], flush=True)
    pooled = pd.concat(outputs, ignore_index=True)
    pooled.to_csv(out / 'pooled_oof.csv', index=False)
    pd.DataFrame(metrics).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(coefficients).to_csv(out / 'coefficients.csv', index=False)
    tn, fp, fn, tp = confusion_matrix(pooled.label, pooled.youden_prediction).ravel()
    summary = dict(seed=args.seed, cv=args.cv, n_splits=len(splits),
        n_subjects=int(df['Study ID'].nunique()), n_subject_sides=len(df),
        pooled_auc=roc_auc_score(pooled.label, pooled.probability),
        sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
        model=f'{args.penalty.upper()} logistic regression, C=1, balanced class weights; no validation-based selection',
        threshold='Training-derived Youden per fold',
        sampling='One record per subject/side', features=features)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
