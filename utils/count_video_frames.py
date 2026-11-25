r"""
Count frames for every video referenced in the combined dataset CSV, and
print summary statistics to help choose `clip_len` / `step_size`.

Usage:
    python -m utils.count_video_frames --csv data/all_data_combined.csv --video_root /path/to/videos

Notes:
- Uses cv2.CAP_PROP_FRAME_COUNT for speed. This is occasionally wrong for
  some codecs/containers (it can be a stale index rather than the true
  frame count), so the script also does a slower manual-count fallback
  you can enable with --verify to catch mismatches.
- Video paths in the CSV are Windows-style relative paths
  (".\Myofascial Pain Ultrasound\...\....avi" or "Preprocess_2\..."). Adjust
  `--video_root` to wherever those paths actually resolve on this machine.
"""
import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm


def resolve_path(video_root, raw_path):
    # CSV paths use Windows separators; normalize for the current OS.
    cleaned = raw_path.replace("\\", os.sep).lstrip(f".{os.sep}")
    return os.path.join(video_root, cleaned)


def count_frames_fast(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def count_frames_manual(path):
    """Slower but exact: actually reads every frame. Use for --verify only."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_data_combined.csv")
    ap.add_argument("--video_root", default=".")
    ap.add_argument("--video_col", default="Video")
    ap.add_argument("--out", default="video_frame_counts.csv")
    ap.add_argument("--verify", action="store_true",
                     help="also do a slow manual frame count to sanity-check the fast metadata count")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    unique_videos = df[[args.video_col, "Study ID", "label"]].drop_duplicates(subset=args.video_col)

    rows = []
    for _, r in tqdm(unique_videos.iterrows(), total=len(unique_videos), desc="counting frames"):
        path = resolve_path(args.video_root, r[args.video_col])
        n_fast = count_frames_fast(path)
        n_manual = count_frames_manual(path) if args.verify else None
        rows.append({
            "video": r[args.video_col],
            "subject": r["Study ID"],
            "label": r["label"],
            "n_frames_metadata": n_fast,
            "n_frames_manual": n_manual,
            "exists": n_fast is not None,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    missing = out_df["exists"].eq(False).sum()
    if missing:
        print(f"WARNING: {missing} video(s) could not be opened -- check --video_root or file paths.")

    valid = out_df[out_df["exists"]]
    print(f"\n{len(valid)} videos read successfully.")
    print(valid["n_frames_metadata"].describe())

    if args.verify:
        mismatch = valid[valid["n_frames_metadata"] != valid["n_frames_manual"]]
        if len(mismatch):
            print(f"\n{len(mismatch)} videos where metadata count != manual count -- "
                  f"metadata frame count may be unreliable for these files:")
            print(mismatch[["video", "n_frames_metadata", "n_frames_manual"]])

    print(f"\nSaved per-video frame counts to {args.out}")


if __name__ == "__main__":
    main()
