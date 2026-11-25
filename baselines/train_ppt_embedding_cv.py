"""Separate pre/post PPT embedding baseline with subject-disjoint five-fold CV."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from train_cv import EMATracker, youden_threshold
from cohort import healthy_latent_cohort
from config import Config


class PPTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(1, 4, bias=False)
        self.post = nn.Linear(1, 4, bias=False)
        self.classifier = nn.Linear(8, 1)

    def forward(self, x):
        return self.classifier(torch.cat([self.pre(x[:, :1]), self.post(x[:, 1:2])], 1)).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    columns = ['PPT_PRE_AVG', 'PPT_POST_AVG']
    source = healthy_latent_cohort(pd.read_csv(Config.data_csv))
    if source.groupby(['Study ID', 'Side'])[columns + ['label']].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting clinical inputs within subject/side')
    df = source.drop_duplicates(['Study ID', 'Side']).copy()
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError('PPT inputs must be finite')
    outputs, metrics = [], []
    for fold in range(1, 6):
        torch.manual_seed(42 + fold - 1)
        train, val = df[df.cv_fold != fold], df[df.cv_fold == fold]
        assert not set(train['Study ID']) & set(val['Study ID'])
        scaler = StandardScaler().fit(train[columns])
        tx = torch.tensor(scaler.transform(train[columns]), dtype=torch.float32)
        vx = torch.tensor(scaler.transform(val[columns]), dtype=torch.float32)
        ty = torch.tensor(train.label.to_numpy(), dtype=torch.float32)
        vy = torch.tensor(val.label.to_numpy(), dtype=torch.float32)
        model = PPTClassifier()
        criterion = nn.BCEWithLogitsLoss(pos_weight=(1-ty).sum()/ty.sum())
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if p.ndim > 1], 'weight_decay': .0003},
            {'params': [p for p in model.parameters() if p.ndim <= 1], 'weight_decay': 0.}], lr=.003)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.01, total_iters=10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[10])
        tracker, history = EMATracker(.8), []
        for epoch in range(40):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(tx), ty)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                logits = model(vx)
                val_loss = criterion(logits, vy).item()
                auc = roc_auc_score(vy.numpy(), logits.sigmoid().numpy())
            el, ea, score = tracker.update(val_loss, auc)
            if tracker.is_best(score):
                best, best_epoch = copy.deepcopy(model.state_dict()), epoch
            history.append(dict(epoch=epoch, train_loss=loss.item(), val_loss=val_loss,
                                auc=auc, ema_loss=el, ema_auc=ea, combined_score=score))
        model.load_state_dict(best)
        with torch.no_grad():
            threshold = youden_threshold(ty.numpy(), model(tx).sigmoid().numpy())
            probability = model(vx).sigmoid().numpy()
        result = val[['Study ID', 'Side', 'label', 'cv_fold']].rename(
            columns={'Study ID': 'subject', 'Side': 'side', 'cv_fold': 'fold'}).copy()
        result['probability'] = probability
        result['youden_threshold'] = threshold
        result['youden_prediction'] = (probability >= threshold).astype(int)
        directory = out/f'fold-{fold}'
        directory.mkdir(exist_ok=True)
        torch.save(dict(state_dict=best, scaler_mean=scaler.mean_.tolist(),
                        scaler_scale=scaler.scale_.tolist(), feature_order=columns,
                        epoch=best_epoch, seed=42+fold-1, youden_threshold=threshold), directory/'best_model.pth')
        pd.DataFrame(history).to_csv(directory/'training_history.csv', index=False)
        result.to_csv(directory/'oof_val_result.csv', index=False)
        outputs.append(result)
        metrics.append(dict(fold=fold, n=len(val), auc=roc_auc_score(val.label, probability),
                            best_epoch=best_epoch, youden_threshold=threshold))
        print(metrics[-1], flush=True)
    pooled = pd.concat(outputs, ignore_index=True)
    pooled.to_csv(out/'pooled_oof.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv', index=False)
    tn, fp, fn, tp = confusion_matrix(pooled.label, pooled.youden_prediction).ravel()
    summary = dict(epochs=40, warmup=10, seed=42, ema_alpha=.8, features=columns,
        n_subjects=int(df['Study ID'].nunique()), n_subject_sides=len(df),
        pooled_auc=roc_auc_score(pooled.label, pooled.probability),
        mean_fold_auc=float(np.mean([m['auc'] for m in metrics])),
        sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
        selection='Outer validation EMA AUC minus loss; not an independent test estimate')
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
