"""
run_fold_eval.py
=================
Consumes the .npy files produced by extract_fold_embeddings.py for ONE fold
and runs the simclr_eval.py quality evaluation three times:

    1. held_out_subjects only  <- the metric that actually matters: this
       fold's encoder never saw these subjects at all during pretraining,
       so this is the genuine test of whether the representation
       generalizes.
    2. train_subjects only     <- upper-bound / sanity check: the encoder
       DID see these subjects (unsupervised) during pretraining.
    3. combined                <- both together, for the overall picture.

Then prints a side-by-side comparison of held-out vs. train metrics. A big
gap (train looks great, held-out looks mediocre) tells you the encoder is
overfitting to subject-specific nuisance texture rather than learning the
anatomical/pathological signal you actually want -- worth knowing before
you spend a finetuning run on it.

Usage
-----
    python -m simclr.fold_evaluate --data_dir ./simclr_eval_data/fold-2 \
        --outdir ./simclr_eval_out/fold-2
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from simclr.evaluate import run_full_evaluation, alignment_uniformity


def load_split(data_dir):
    emb = np.load(os.path.join(data_dir, "embeddings.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(data_dir, "groups.npy"), allow_pickle=True)
    splits = np.load(os.path.join(data_dir, "splits.npy"), allow_pickle=True)
    z1_path = os.path.join(data_dir, "z_view1.npy")
    z2_path = os.path.join(data_dir, "z_view2.npy")
    z1 = np.load(z1_path) if os.path.exists(z1_path) else None
    z2 = np.load(z2_path) if os.path.exists(z2_path) else None
    return emb, labels, groups, splits, z1, z2


def subset(mask, emb, labels, groups, z1, z2):
    e = emb[mask]
    l = labels[mask]
    g = groups[mask]
    p1 = z1[mask] if z1 is not None else None
    p2 = z2[mask] if z2 is not None else None
    return e, l, g, p1, p2


def plot_comparison(held_report, train_report, outpath):
    metrics = []
    held_vals, train_vals = [], []

    def add(name, h_val, t_val):
        if h_val is not None and t_val is not None:
            metrics.append(name)
            held_vals.append(h_val)
            train_vals.append(t_val)

    if held_report.get("linear_probe") and train_report.get("linear_probe"):
        for key in ["accuracy", "balanced_accuracy", "mcc", "auc"]:
            if key in held_report["linear_probe"] and key in train_report["linear_probe"]:
                add(f"linear_probe_{key}", held_report["linear_probe"][key][0], train_report["linear_probe"][key][0])

    if held_report.get("clustering_quality") and train_report.get("clustering_quality"):
        if held_report["clustering_quality"]["silhouette"] is not None:
            add("silhouette", held_report["clustering_quality"]["silhouette"], train_report["clustering_quality"]["silhouette"])

    if not metrics:
        return

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.3), 4.5))
    ax.bar(x - width / 2, held_vals, width, label="held_out_subjects", color="tab:red")
    ax.bar(x + width / 2, train_vals, width, label="train_subjects", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Held-out vs. train-subject representation quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="output dir from extract_fold_embeddings.py")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--proj_method", choices=["tsne", "umap"], default="tsne")
    args = parser.parse_args()

    emb, labels, groups, splits, z1, z2 = load_split(args.data_dir)

    held_mask = splits == "held_out_subjects"
    train_mask = splits == "train_subjects"
    print(f"held_out_subjects clips: {held_mask.sum()}   train_subjects clips: {train_mask.sum()}")

    os.makedirs(args.outdir, exist_ok=True)

    # NOTE: alignment/uniformity (Wang & Isola) must be computed in the SAME
    # space for both arguments -- z_view1 vs z_view2 (projector output), not
    # h vs z (different dimensionality, different space). run_full_evaluation's
    # `pos_pairs` arg is therefore only meaningful when `embeddings` is itself
    # in z-space, so here we compute it separately and attach it to each report
    # rather than passing z1 in as if it matched the h-space embeddings.

    def add_alignment(report, p1, p2, tag):
        if p1 is not None and p2 is not None and len(p1) > 1:
            align, unif = alignment_uniformity(p1, p2)
            report["alignment_uniformity"] = {"alignment": align, "uniformity": unif}
            print(f"[{tag}] alignment={align:.4f} uniformity={unif:.4f}  (projector z-space, lower=better for both)")
        return report

    print("\n" + "#" * 70)
    print("# HELD-OUT SUBJECTS  (the genuine generalization test)")
    print("#" * 70)
    e, l, g, p1, p2 = subset(held_mask, emb, labels, groups, z1, z2)
    n_splits_held = min(args.n_splits, len(np.unique(g)) if len(np.unique(l)) > 1 else args.n_splits)
    held_report = run_full_evaluation(
        e, l, g, None, outdir=os.path.join(args.outdir, "held_out_subjects"),
        n_splits=max(2, n_splits_held), proj_method=args.proj_method)
    held_report = add_alignment(held_report, p1, p2, "held_out")

    print("\n" + "#" * 70)
    print("# TRAIN SUBJECTS  (seen unsupervised during pretraining)")
    print("#" * 70)
    e, l, g, p1, p2 = subset(train_mask, emb, labels, groups, z1, z2)
    n_splits_train = min(args.n_splits, len(np.unique(g)) if len(np.unique(l)) > 1 else args.n_splits)
    train_report = run_full_evaluation(
        e, l, g, None, outdir=os.path.join(args.outdir, "train_subjects"),
        n_splits=max(2, n_splits_train), proj_method=args.proj_method)
    train_report = add_alignment(train_report, p1, p2, "train")

    print("\n" + "#" * 70)
    print("# COMBINED (both splits together)")
    print("#" * 70)
    combined_report = run_full_evaluation(
        emb, labels, groups, None, outdir=os.path.join(args.outdir, "combined"),
        n_splits=args.n_splits, proj_method=args.proj_method)
    combined_report = add_alignment(combined_report, z1, z2, "combined")

    plot_comparison(held_report, train_report, os.path.join(args.outdir, "held_vs_train_comparison.png"))

    with open(os.path.join(args.outdir, "held_vs_train_summary.json"), "w") as f:
        json.dump({"held_out_subjects": held_report, "train_subjects": train_report}, f, indent=2, default=str)

    print(f"\nAll reports + comparison plot saved under: {os.path.abspath(args.outdir)}")
    print("Look specifically at held_out_subjects/report.json vs. train_subjects/report.json -- "
          "a big gap there (train much better than held-out) suggests the encoder is picking up "
          "subject-specific nuisance signal rather than the anatomical/pathological pattern you want.")
