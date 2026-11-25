# Data

**No patient data is distributed with this repository.** Everything in this
directory except this file and `metadata_template.csv` is git-ignored. Do not
commit videos, frames, clinical metadata, or anything derived from them.
Use of clinical data must follow your own ethics approval (IRB/REB) and
data-use agreements.

## Expected layout

```text
data/
  all_data_combined.csv   # metadata table (one row per video), see schema below
  videos/...              # ultrasound clips; any layout, referenced by the Video column
```

The path is set by `Config.data_csv` in `config.py`.

## Metadata schema

`metadata_template.csv` has the header row. Columns:

| Column | Type | Used by | Description |
|---|---|---|---|
| `Study ID` | str | all | Pseudonymous subject ID (e.g. `SUBJ01`). Never use names or MRNs. Folds are split by subject. |
| `Side` | str | all | Scanned side, e.g. `L` / `R`. |
| `MTrP type` | str | all | `Healthy` or `Latent`. Other values are filtered out by `cohort.py`. |
| `label` | int | all | `0` = Healthy, `1` = Latent. Must agree with `MTrP type`. |
| `Video` | str | all | Path to the video file (relative to the repo root, or absolute). |
| `Intervention` | str | all | Acquisition time point, e.g. `Pre` / `Post`. |
| `cv_fold` | int | all | Outer CV fold, 1..`n_fold`. Produced by `utils/build_folds.py`. |
| `source_file` | str | fold building | Acquisition batch or source label, used to stratify folds. |
| `Sex` | str | `--use_demo` | `M` / `F`. |
| `Age` | float | `--use_demo` | Years. Standardized per fold on the training split only. |
| `PPT_PRE_AVG`, `PPT_POST_AVG` | float | `--use_demo` | Pressure-pain-threshold averages. Standardized per fold. |

`cohort.py` validates the table before training. It rejects rows with missing
values, a subject that appears in more than one fold, conflicting labels for
the same subject and side, duplicate videos, and folds that lack one of the
two classes.

## Video format

Any format OpenCV can decode (`.avi`, `.mp4`, ...). Frames are converted to
8-bit grayscale on load. Crop each video to the B-mode imaging region
beforehand: remove scanner UI, burned-in text, and anything that could
identify a patient. The pipeline does not crop or de-identify.
