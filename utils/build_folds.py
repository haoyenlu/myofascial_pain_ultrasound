"""
Combine one or more metadata CSVs into one file and rebuild the
5-fold split at the SUBJECT level, stratified by (source_file, label).

Why rebuild the folds instead of reusing the old cv_fold column:
- The old folds ranged from 3 to 13 subjects each -- far too imbalanced,
  and were the main driver of the huge fold-to-fold AUC swing we saw
  (0.26 to 0.94 on the same architecture/hyperparameters).
- The old split wasn't stratified by data source. Two source CSVs were
  merged (`all_new_data_2.csv`, `data_restructured_latent.csv`), and one
  fold ended up 100% one source -- confounding "bad fold" with "bad source".
- We're also folding the old 5-subject holdout back into the pool. With
  only ~38-44 subjects total, a fixed tiny holdout is too noisy to trust
  on its own (we saw subject-level AUC = 0.50 on it with only 1 negative
  subject). Pooled 5-fold out-of-fold (OOF) predictions across all subjects
  give a far more stable evaluation with no subject wasted.

Splitting is done with sklearn's StratifiedGroupKFold:
- `groups=Study ID` guarantees no subject appears in more than one fold
  (this is what prevents leakage -- a subject's left/right-side videos,
  which can carry different labels, always land in the same fold).
- `y=strata` (source_file + majority label) tries to keep both the
  class balance AND the source-mix balance even across folds.

Note: a couple of subjects have BOTH labels across their rows (different
sides scanned as Healthy vs Latent independently). That's expected, not a
bug -- we stratify by each subject's *majority* label since the fold
assignment has to be one decision per subject.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

OUT_PATH = "data/all_data_combined.csv"
N_SPLITS = 5
RANDOM_STATE = 42


def build_folds(raw_files, out_path=OUT_PATH, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    combined = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)

    # one row per subject: majority label + source, used only for stratification
    subj = combined.groupby("Study ID").agg(
        label_majority=("label", lambda x: x.mode()[0]),
        source=("source_file", "first"),
    ).reset_index()
    subj["strata"] = subj["source"].astype(str) + "_" + subj["label_majority"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    subj["cv_fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(subj, subj["strata"], groups=subj["Study ID"]), start=1
    ):
        subj.loc[val_idx, "cv_fold"] = fold_idx

    print("Subjects per fold:")
    print(subj.groupby("cv_fold").agg(n_subj=("Study ID", "size"), n_pos=("label_majority", "sum")))
    print("\nSource mix per fold:")
    print(pd.crosstab(subj["cv_fold"], subj["source"]))

    merged = combined.drop(columns=["split_set", "cv_fold"], errors="ignore").merge(
        subj[["Study ID", "cv_fold"]], on="Study ID"
    )
    merged["split_set"] = "cv"

    print("\nVideo-level rows per fold:")
    print(merged["cv_fold"].value_counts().sort_index())

    merged.to_csv(out_path, index=False)
    print(f"\nSaved combined, re-folded dataset to {out_path}")
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge metadata CSVs and assign subject-level CV folds.")
    parser.add_argument("inputs", nargs="+", help="One or more metadata CSVs (see data/README.md for the schema).")
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    build_folds(args.inputs, args.out, args.n_splits, args.seed)
