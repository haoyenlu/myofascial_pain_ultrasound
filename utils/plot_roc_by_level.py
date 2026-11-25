r"""
Plot ROC curves at three levels of aggregation -- sample (clip), video, and
subject -- from an OOF validation result CSV.

Expected CSV columns: subject, label, probability, video_id
(an optional `fold` column is ignored for plotting, but is fine to have --
useful if you want to filter to one fold before calling this).

Usage:
    # single run
    python -m utils.plot_roc_by_level --csv oof_val_result.csv --out roc.png

    # any number of runs overlaid on the same plot (e.g. fold 1 through 5)
    python -m utils.plot_roc_by_level \
        --csv fold1.csv fold2.csv fold3.csv fold4.csv fold5.csv \
        --labels "Fold 1" "Fold 2" "Fold 3" "Fold 4" "Fold 5" \
        --out roc_all_folds.png --title "Per-fold OOF ROC curves"

    # labels are optional -- defaults to "Run 1", "Run 2", ... if omitted
    python -m utils.plot_roc_by_level --csv fold1.csv fold2.csv --out roc.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def aggregate_levels(df):
    """
    Returns dict: level -> (y, p, n)
    - sample: every row as-is (one prediction per clip)
    - video: mean probability per video_id
    - subject: mean probability per subject
    """
    vid = df.groupby("video_id").agg(mean_prob=("probability", "mean"), label=("label", "first"))
    subj = df.groupby("subject").agg(mean_prob=("probability", "mean"), label=("label", "first"))
    return {
        "sample": (df["label"].values, df["probability"].values, len(df)),
        "video": (vid["label"].values, vid["mean_prob"].values, len(vid)),
        "subject": (subj["label"].values, subj["mean_prob"].values, len(subj)),
    }


def plot_roc_by_level(csv_paths_and_labels, out_path, title=None, colors=None):
    """
    csv_paths_and_labels: list of (csv_path, label) tuples -- one entry for a
                          single run, two+ entries to overlay multiple runs
                          for comparison (e.g. different clip lengths).
    """
    default_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a6559c"]
    if colors is None:
        colors = default_colors

    level_titles = {"sample": "Sample-level (per clip)", "video": "Video-level", "subject": "Subject-level"}
    levels = ["sample", "video", "subject"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for run_idx, (csv_path, label) in enumerate(csv_paths_and_labels):
        df = pd.read_csv(csv_path)
        agg = aggregate_levels(df)
        color = colors[run_idx % len(colors)]

        for ax, level in zip(axes, levels):
            y, p, n = agg[level]
            auc = roc_auc_score(y, p)
            fpr, tpr, _ = roc_curve(y, p)
            legend_label = f"{label} (AUC={auc:.3f}, n={n})" if label else f"AUC={auc:.3f}, n={n}"
            if level == "subject":
                # small n -> a smooth line is misleading; step plot is honest
                # about how coarse the curve actually is
                ax.step(fpr, tpr, where="post", color=color, lw=2.2, label=legend_label)
            else:
                ax.plot(fpr, tpr, color=color, lw=2.2, label=legend_label)

    for ax, level in zip(axes, levels):
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(level_titles[level])
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, nargs="+",
                         help="One or more OOF validation result CSVs (e.g. --csv fold1.csv fold2.csv fold3.csv)")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Legend label for each CSV, same order/count as --csv "
                              "(e.g. --labels 'Fold 1' 'Fold 2' 'Fold 3'). "
                              "If omitted, defaults to 'Run 1', 'Run 2', ...")
    parser.add_argument("--out", default="roc_by_level.png", help="Output PNG path")
    parser.add_argument("--title", default=None, help="Overall figure title")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.csv):
        parser.error(f"--labels has {len(args.labels)} entries but --csv has {len(args.csv)} -- must match")

    labels = args.labels if args.labels is not None else [f"Run {i+1}" for i in range(len(args.csv))]
    runs = list(zip(args.csv, labels))

    plot_roc_by_level(runs, args.out, title=args.title)
