r"""
visualize_fold_embeddings.py
=============================
Companion to extract_fold_embeddings.py. Loads the .npy files it wrote
to --outdir and produces the standard SimCLR representation-quality
visualizations:

  1. 2D projection (UMAP if installed, else t-SNE) of the encoder
     features h, colored three ways: by label, by train/held-out split,
     and by subject -- these are three different questions, not three
     views of the same fact:
       - colored by label:   does the (unsupervised) encoder already
                              produce class-separable features, before
                              any finetuning touches them?
       - colored by split:   do held_out_subjects' embeddings land in
                              the same region as train_subjects', or
                              form their own separate island? The latter
                              is evidence the encoder doesn't generalize
                              to genuinely unseen subjects.
       - colored by subject: does the embedding cluster by WHO the
                              subject is rather than by class? This is
                              the classic failure mode for medical video
                              SimCLR -- learning probe angle/anatomy/
                              equipment identity instead of anything
                              about the condition. If this plot shows
                              tight per-subject clusters that don't
                              correspond to the label plot's clusters,
                              that's a real problem, not a stylistic
                              choice of colormap.

  2. If z_view1.npy/z_view2.npy exist (i.e. extract_fold_embeddings.py
     was run with --extract_pairs): alignment and uniformity (Wang &
     Isola 2020) -- a numeric, non-visual complement to the 2D
     projections above, computed directly in the space NT-Xent actually
     optimizes (the projector output z), not the encoder features h.

Usage
-----
    python -m simclr.visualize_embeddings --data_dir ./simclr_eval_data/fold-2
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# 2D projection: UMAP if available (faster, generally better global
# structure for this kind of check), falling back to t-SNE. Both are run
# on a PCA-reduced version of h first -- standard practice for
# high-dimensional (2048-d) input, and considerably faster.
# --------------------------------------------------------------------------- #

def compute_2d_projection(h, pca_dim=50, seed=0):
    from sklearn.decomposition import PCA

    n_components = min(pca_dim, h.shape[0] - 1, h.shape[1])
    h_reduced = PCA(n_components=n_components, random_state=seed).fit_transform(h)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords = reducer.fit_transform(h_reduced)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        print("[info] umap-learn not installed; falling back to t-SNE "
              "(pip install umap-learn for faster/often cleaner projections)")
        coords = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(h_reduced)
        method = "t-SNE"

    return coords, method


def plot_colored_scatter(ax, coords, color_values, title, categorical=True, cmap="tab20"):
    if categorical:
        unique_vals = sorted(set(color_values))
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        color_idx = np.array([val_to_idx[v] for v in color_values])
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_idx, cmap=cmap,
                              s=8, alpha=0.7, linewidths=0)
        # legend only if few enough categories to be readable
        if len(unique_vals) <= 12:
            handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                    color=plt.get_cmap(cmap)(i / max(len(unique_vals) - 1, 1)))
                       for i in range(len(unique_vals))]
            ax.legend(handles, unique_vals, fontsize=7, loc="best", ncol=1 if len(unique_vals) <= 6 else 2)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, cmap=cmap, s=8, alpha=0.7)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------- #
# Alignment / uniformity (Wang & Isola 2020), computed on the projector
# output z (already L2-normalized, per extract_fold_embeddings.py).
# --------------------------------------------------------------------------- #

def alignment_loss(z1, z2, alpha=2):
    """Mean ||z1-z2||^alpha over positive pairs (the two augmented views
    of the SAME clip). Lower = the two views of a clip land closer
    together -- the property NT-Xent's positive-pair term optimizes for."""
    dist = np.linalg.norm(z1 - z2, axis=1)
    return np.mean(dist ** alpha)


def uniformity_loss(z, t=2, max_pairs=200_000, seed=0):
    """log E[exp(-t * ||zi-zj||^2)] over pairs of DIFFERENT points.
    Lower (more negative) = embeddings are more spread out over the
    hypersphere, rather than collapsed into a small region -- the
    property NT-Xent's negative-pair term optimizes for. Subsamples
    pairs if there are too many points to do this exactly in memory."""
    n = z.shape[0]
    rng = np.random.RandomState(seed)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.randint(0, n, size=max_pairs)
        j = rng.randint(0, n, size=max_pairs)
        mask = i != j
        i, j = i[mask], j[mask]
    sq_dist = np.sum((z[i] - z[j]) ** 2, axis=1)
    return np.log(np.mean(np.exp(-t * sq_dist)))


def plot_alignment_uniformity(z1, z2, outpath):
    dist = np.linalg.norm(z1 - z2, axis=1)
    align = alignment_loss(z1, z2)
    unif = uniformity_loss(np.concatenate([z1, z2], axis=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dist, bins=40, color="#2954E0", alpha=0.75)
    ax.axvline(dist.mean(), color="#E63946", ls="--", lw=2,
               label=f"mean = {dist.mean():.3f}")
    ax.set_xlabel("||z_view1 - z_view2|| (positive pairs -- same clip, two augmentations)")
    ax.set_ylabel("count")
    ax.set_title(f"Alignment: {align:.4f}  |  Uniformity: {unif:.4f}\n"
                 f"(lower alignment = better; more negative uniformity = better spread)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[info] alignment={align:.4f}  uniformity={unif:.4f}  (saved {outpath})")
    return align, unif


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(data_dir, outdir=None, max_points=5000, seed=0):
    outdir = outdir or data_dir
    os.makedirs(outdir, exist_ok=True)

    h = np.load(os.path.join(data_dir, "embeddings.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(data_dir, "groups.npy"), allow_pickle=True)
    splits = np.load(os.path.join(data_dir, "splits.npy"), allow_pickle=True)
    print(f"[info] loaded {h.shape[0]} embeddings, dim={h.shape[1]}")
    print(f"[info] splits: {dict(zip(*np.unique(splits, return_counts=True)))}")
    print(f"[info] unique subjects/groups: {len(set(groups))}")

    # Subsample for the 2D projection if there are a lot of clips --
    # t-SNE/UMAP runtime and plot readability both degrade badly past a
    # few thousand points; this doesn't affect the alignment/uniformity
    # numbers below, which use the full data.
    n = h.shape[0]
    if n > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_points, replace=False)
        print(f"[info] subsampling {max_points}/{n} points for the 2D projection "
              f"(pass --max_points to change)")
    else:
        idx = np.arange(n)

    coords, method = compute_2d_projection(h[idx], seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    plot_colored_scatter(axes[0], coords, labels[idx], f"Colored by label ({method})", cmap="coolwarm")
    plot_colored_scatter(axes[1], coords, splits[idx], f"Colored by split ({method})", cmap="Set1")
    plot_colored_scatter(axes[2], coords, groups[idx], f"Colored by subject ({method})", cmap="tab20")
    fig.suptitle(f"Fold embedding quality check -- {os.path.basename(os.path.normpath(data_dir))}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    scatter_path = os.path.join(outdir, "embedding_projection.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[info] saved {scatter_path}")

    # Alignment/uniformity, only if the paired projector embeddings exist
    z1_path = os.path.join(data_dir, "z_view1.npy")
    z2_path = os.path.join(data_dir, "z_view2.npy")
    if os.path.exists(z1_path) and os.path.exists(z2_path):
        z1, z2 = np.load(z1_path), np.load(z2_path)
        plot_alignment_uniformity(z1, z2, os.path.join(outdir, "alignment_uniformity.png"))
    else:
        print("[info] z_view1.npy/z_view2.npy not found -- skipping alignment/uniformity "
              "(re-run extract_fold_embeddings.py with --extract_pairs to get these)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True,
                         help="Directory written by extract_fold_embeddings.py's --outdir")
    parser.add_argument("--outdir", default=None,
                         help="Where to save plots. Defaults to --data_dir.")
    parser.add_argument("--max_points", type=int, default=5000,
                         help="Subsample the 2D projection to at most this many points "
                              "(t-SNE/UMAP runtime and plot readability both degrade past "
                              "a few thousand). Alignment/uniformity always use full data.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.data_dir, args.outdir, args.max_points, args.seed)
