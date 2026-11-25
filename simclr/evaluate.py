"""
simclr_eval.py
==============
Evaluate and visualize the quality of SimCLR-learned embeddings extracted
from ultrasound video clips (e.g. MTrP Healthy vs. Latent classification).

Inputs expected as .npy files (or pass numpy arrays directly if importing
this as a module instead of running from the CLI):

    embeddings.npy   -> shape (N, D)   float32/float64, one row per clip
    labels.npy       -> shape (N,)     int/str class labels (optional but
                                        needed for most metrics/plots)
    groups.npy       -> shape (N,)     subject/source IDs (optional, used
                                        for subject-level grouped CV so a
                                        subject never appears in both train
                                        and val — same logic as your
                                        StratifiedGroupKFold fold rebuild)
    pos_pairs.npy    -> shape (N, D)   optional. If your embeddings came in
                                        augmented pairs (z_i, z_j), pass the
                                        SECOND view here (same subject/clip
                                        order as embeddings.npy) to unlock
                                        alignment/uniformity metrics
                                        (Wang & Isola, 2020).

What this produces
-------------------
1. Collapse / representation-health diagnostics
   - embedding norm distribution
   - singular value spectrum + effective rank (99% variance)
2. Alignment & uniformity (only if pos_pairs given) — the standard SimCLR
   representation-quality metrics from Wang & Isola 2020
3. Discriminability of the representation w.r.t. labels
   - kNN accuracy vs. k, with subject-grouped CV
   - linear probe (logistic regression) AUC/accuracy, subject-grouped CV
   - silhouette / Davies-Bouldin / Calinski-Harabasz w.r.t. true labels
4. Visualizations (saved as PNGs)
   - PCA scree plot + effective rank
   - cosine-similarity heatmap, samples sorted by label
   - t-SNE (and UMAP if installed) 2D projection colored by label
   - kNN accuracy vs. k curve
   - per-sample silhouette plot

Usage
-----
    python -m simclr.evaluate \
        --embeddings embeddings.npy \
        --labels labels.npy \
        --groups groups.npy \
        --outdir ./simclr_eval_out

Or import and call `run_full_evaluation(...)` directly from a notebook.
"""

import argparse
import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import umap  # noqa: F401
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# 1. Collapse / representation-health diagnostics
# --------------------------------------------------------------------------- #

def effective_rank(embeddings, variance_threshold=0.99):
    """Number of PCA components needed to explain `variance_threshold` of
    variance. A representation that has collapsed onto a low-dimensional
    manifold (or a single point) will have a very low effective rank
    relative to the embedding dimension D — a classic SimCLR failure mode.
    """
    emb_centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD is more numerically stable than eig on the covariance matrix
    _, s, _ = np.linalg.svd(emb_centered, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    cumulative = np.cumsum(explained)
    eff_rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return eff_rank, s, explained


def norm_stats(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_cv": float(norms.std() / (norms.mean() + 1e-12)),  # coefficient of variation
    }, norms


# --------------------------------------------------------------------------- #
# 2. Alignment & uniformity (Wang & Isola, 2020)
# --------------------------------------------------------------------------- #

def alignment_uniformity(z1, z2, t=2.0):
    """
    z1, z2: (N, D) matched positive-pair embeddings (already L2-normalized
            internally). Lower alignment = positive pairs sit closer together
            (good). Lower uniformity = embeddings spread more evenly over the
            hypersphere (good, indicates less collapse).
    Reference: Wang & Isola, "Understanding Contrastive Representation
    Learning through Alignment and Uniformity on the Hypersphere", ICML 2020.
    """
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12)

    alignment = np.mean(np.sum((z1 - z2) ** 2, axis=1))

    def _uniformity(z):
        sq_dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        n = z.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return np.log(np.mean(np.exp(-t * sq_dists[mask])))

    uniformity = 0.5 * (_uniformity(z1) + _uniformity(z2))
    return float(alignment), float(uniformity)


# --------------------------------------------------------------------------- #
# 3. Discriminability w.r.t. labels
# --------------------------------------------------------------------------- #

def _make_cv(labels, groups, n_splits=5, seed=42):
    """Subject-grouped CV when groups are provided (mirrors your
    StratifiedGroupKFold fold-rebuild logic), else plain StratifiedKFold."""
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(labels)), labels))
    return splits


def knn_eval(embeddings, labels, groups=None, k_values=(1, 3, 5, 10, 15, 25), n_splits=5, seed=42):
    """kNN accuracy/AUC vs. k using subject-grouped CV. A representation
    where same-class points are close together will show high kNN accuracy
    even with a very simple, non-parametric classifier — this is the
    standard 'representation quality' proxy used in SimCLR papers."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    splits = _make_cv(labels, groups, n_splits, seed)

    results = {k: {"acc": [], "bacc": [], "mcc": [], "auc": []} for k in k_values}
    for train_idx, val_idx in splits:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]
        for k in k_values:
            if k > len(train_idx):
                continue
            clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_val)
            results[k]["acc"].append((pred == y_val).mean())
            results[k]["bacc"].append(balanced_accuracy_score(y_val, pred))
            results[k]["mcc"].append(matthews_corrcoef(y_val, pred))
            if len(np.unique(labels)) == 2:
                proba = clf.predict_proba(X_val)[:, 1]
                try:
                    results[k]["auc"].append(roc_auc_score(y_val, proba))
                except ValueError:
                    pass  # only one class present in this fold's val set

    summary = {}
    for k, metrics in results.items():
        summary[k] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items() if len(v) > 0}
    return summary


def linear_probe_eval(embeddings, labels, groups=None, n_splits=5, seed=42, max_iter=2000):
    """Logistic regression linear probe — the most common way papers report
    'is this representation linearly separable by class'."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    splits = _make_cv(labels, groups, n_splits, seed)

    accs, baccs, mccs, aucs = [], [], [], []
    for train_idx, val_idx in splits:
        clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        clf.fit(X[train_idx], labels[train_idx])
        pred = clf.predict(X[val_idx])
        accs.append((pred == labels[val_idx]).mean())
        baccs.append(balanced_accuracy_score(labels[val_idx], pred))
        mccs.append(matthews_corrcoef(labels[val_idx], pred))
        if len(np.unique(labels)) == 2:
            proba = clf.predict_proba(X[val_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(labels[val_idx], proba))
            except ValueError:
                pass

    out = {
        "accuracy": (float(np.mean(accs)), float(np.std(accs))),
        "balanced_accuracy": (float(np.mean(baccs)), float(np.std(baccs))),
        "mcc": (float(np.mean(mccs)), float(np.std(mccs))),
    }
    if aucs:
        out["auc"] = (float(np.mean(aucs)), float(np.std(aucs)))
    return out


def clustering_quality(embeddings, labels):
    """How well the *true* labels correspond to geometric clusters in the
    embedding space. These don't require training a classifier at all."""
    out = {}
    try:
        out["silhouette"] = float(silhouette_score(embeddings, labels, metric="cosine"))
    except ValueError:
        out["silhouette"] = None
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
    except ValueError:
        out["davies_bouldin"] = None
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
    except ValueError:
        out["calinski_harabasz"] = None
    return out


# --------------------------------------------------------------------------- #
# 4. Visualizations
# --------------------------------------------------------------------------- #

def plot_scree(explained, eff_rank, outpath):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    n_show = min(50, len(explained))
    ax[0].bar(range(1, n_show + 1), explained[:n_show])
    ax[0].set_xlabel("Principal component")
    ax[0].set_ylabel("Explained variance ratio")
    ax[0].set_title("Singular value spectrum")

    cumulative = np.cumsum(explained)
    ax[1].plot(range(1, len(cumulative) + 1), cumulative)
    ax[1].axhline(0.99, color="red", linestyle="--", linewidth=1)
    ax[1].axvline(eff_rank, color="red", linestyle="--", linewidth=1)
    ax[1].set_xlabel("Number of components")
    ax[1].set_ylabel("Cumulative explained variance")
    ax[1].set_title(f"Effective rank @99% var = {eff_rank}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_similarity_heatmap(embeddings, labels, outpath, max_n=300):
    """Cosine similarity matrix, samples sorted by label. Well-separated
    classes show up as distinct light/dark blocks."""
    if len(embeddings) > max_n:
        idx = np.random.RandomState(0).choice(len(embeddings), max_n, replace=False)
        emb, lab = embeddings[idx], labels[idx]
    else:
        emb, lab = embeddings, labels

    order = np.argsort(lab)
    emb_sorted = emb[order]
    norm = emb_sorted / (np.linalg.norm(emb_sorted, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Cosine similarity (sorted by label)")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_projection(embeddings, labels, outpath, method="tsne", seed=42):
    if method == "umap" and HAS_UMAP:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
        title = "UMAP projection"
    else:
        method = "tsne"
        perplexity = min(30, max(5, len(embeddings) // 4))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
        title = "t-SNE projection"

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    le = LabelEncoder()
    lab_codes = le.fit_transform(labels)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=lab_codes, cmap="tab10", s=18, alpha=0.8)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, le.classes_, title="label", loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return method


def plot_knn_curve(knn_summary, outpath):
    ks = sorted(knn_summary.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric, color in zip(["acc", "bacc", "auc"], ["tab:blue", "tab:orange", "tab:green"]):
        means = [knn_summary[k][metric][0] for k in ks if metric in knn_summary[k]]
        stds = [knn_summary[k][metric][1] for k in ks if metric in knn_summary[k]]
        ks_present = [k for k in ks if metric in knn_summary[k]]
        if means:
            ax.errorbar(ks_present, means, yerr=stds, marker="o", label=metric, color=color)
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("score")
    ax.set_title("kNN evaluation vs. k (subject-grouped CV)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_silhouette(embeddings, labels, outpath):
    sil_vals = silhouette_samples(embeddings, labels, metric="cosine")
    order = np.argsort(labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(sil_vals)), sil_vals[order], width=1.0,
           color=["tab:blue" if l == np.unique(labels)[0] else "tab:orange" for l in np.array(labels)[order]])
    ax.axhline(np.mean(sil_vals), color="red", linestyle="--", label=f"mean = {np.mean(sil_vals):.3f}")
    ax.set_xlabel("sample (sorted by label)")
    ax.set_ylabel("silhouette value")
    ax.set_title("Per-sample silhouette (cosine)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_full_evaluation(embeddings, labels=None, groups=None, pos_pairs=None,
                         outdir="./simclr_eval_out", n_splits=5, seed=42,
                         proj_method="tsne"):
    os.makedirs(outdir, exist_ok=True)
    report = {}

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if labels is not None:
        labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)

    # 1. Collapse diagnostics -------------------------------------------------
    eff_rank, singular_values, explained = effective_rank(embeddings)
    n_stats, norms = norm_stats(embeddings)
    report["representation_health"] = {
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "effective_rank_99pct_var": eff_rank,
        **n_stats,
    }
    plot_scree(explained, eff_rank, os.path.join(outdir, "01_scree_effective_rank.png"))

    # 2. Alignment / uniformity ------------------------------------------------
    if pos_pairs is not None:
        pos_pairs = np.asarray(pos_pairs, dtype=np.float64)
        align, unif = alignment_uniformity(embeddings, pos_pairs)
        report["alignment_uniformity"] = {"alignment": align, "uniformity": unif}
    else:
        report["alignment_uniformity"] = None

    # 3 & 4. Label-dependent metrics + plots -----------------------------------
    if labels is not None:
        report["clustering_quality"] = clustering_quality(embeddings, labels)

        knn_summary = knn_eval(embeddings, labels, groups, n_splits=n_splits, seed=seed)
        report["knn_eval"] = knn_summary
        plot_knn_curve(knn_summary, os.path.join(outdir, "04_knn_vs_k.png"))

        report["linear_probe"] = linear_probe_eval(embeddings, labels, groups, n_splits=n_splits, seed=seed)

        plot_similarity_heatmap(embeddings, labels, os.path.join(outdir, "02_cosine_similarity_heatmap.png"))
        used_method = plot_projection(embeddings, labels, os.path.join(outdir, f"03_{proj_method}_projection.png"), method=proj_method)
        report["projection_method_used"] = used_method
        plot_silhouette(embeddings, labels, os.path.join(outdir, "05_silhouette.png"))
    else:
        report["clustering_quality"] = None
        report["knn_eval"] = None
        report["linear_probe"] = None

    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    _print_report(report)
    print(f"\nAll figures + report.json saved to: {os.path.abspath(outdir)}")
    return report


def _print_report(report):
    print("=" * 70)
    print("SimCLR EMBEDDING QUALITY REPORT")
    print("=" * 70)

    rh = report["representation_health"]
    print(f"\n[Representation health]")
    print(f"  N = {rh['n_samples']}, D = {rh['embedding_dim']}")
    print(f"  Effective rank (99% var): {rh['effective_rank_99pct_var']} / {rh['embedding_dim']}")
    print(f"  Norm: mean={rh['norm_mean']:.3f} std={rh['norm_std']:.3f} "
          f"(CV={rh['norm_cv']:.3f}; near 0 CV + low effective rank => possible collapse)")

    if report["alignment_uniformity"] is not None:
        au = report["alignment_uniformity"]
        print(f"\n[Alignment / Uniformity]  (lower = better for both)")
        print(f"  alignment  = {au['alignment']:.4f}")
        print(f"  uniformity = {au['uniformity']:.4f}")

    if report["clustering_quality"] is not None:
        cq = report["clustering_quality"]
        print(f"\n[Clustering quality vs. true labels]")
        print(f"  silhouette (cosine)   = {cq['silhouette']}")
        print(f"  davies-bouldin (lower better) = {cq['davies_bouldin']}")
        print(f"  calinski-harabasz (higher better) = {cq['calinski_harabasz']}")

    if report["knn_eval"] is not None:
        print(f"\n[kNN evaluation, subject-grouped CV]")
        for k, m in sorted(report["knn_eval"].items()):
            parts = [f"{name}={mean:.3f}±{std:.3f}" for name, (mean, std) in m.items()]
            print(f"  k={k:>3}: " + ", ".join(parts))

    if report["linear_probe"] is not None:
        lp = report["linear_probe"]
        print(f"\n[Linear probe (logistic regression), subject-grouped CV]")
        for name, (mean, std) in lp.items():
            print(f"  {name} = {mean:.3f} ± {std:.3f}")

    print("=" * 70)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate & visualize SimCLR embedding quality")
    parser.add_argument("--embeddings", required=True, help="path to embeddings .npy (N, D)")
    parser.add_argument("--labels", default=None, help="path to labels .npy (N,)")
    parser.add_argument("--groups", default=None, help="path to subject/group IDs .npy (N,)")
    parser.add_argument("--pos_pairs", default=None, help="path to second-view embeddings .npy (N, D)")
    parser.add_argument("--outdir", default="./simclr_eval_out")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_method", choices=["tsne", "umap"], default="tsne")
    args = parser.parse_args()

    emb = np.load(args.embeddings)
    lab = np.load(args.labels, allow_pickle=True) if args.labels else None
    grp = np.load(args.groups, allow_pickle=True) if args.groups else None
    pp = np.load(args.pos_pairs) if args.pos_pairs else None

    run_full_evaluation(emb, lab, grp, pp, outdir=args.outdir,
                         n_splits=args.n_splits, seed=args.seed,
                         proj_method=args.proj_method)
