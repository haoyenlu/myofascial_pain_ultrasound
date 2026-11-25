#!/usr/bin/env bash
#
# run_vde_sweep.sh
# =================
# Trains the VDE model once per (timestep, block_to_use) combination,
# using train_cv.py's existing --timestep/--block_to_use CLI overrides
# (see apply_vde_overrides in train_cv.py -- no code changes were needed
# for this, that mechanism already exists and auto-names each
# combination's checkpoint_dir as ./checkpoints/vde_cv_t{timestep}_block{block},
# so different combinations never collide or overwrite each other).
#
# COST WARNING: this is a full grid sweep -- every (timestep, block)
# pair below runs a COMPLETE 5-fold, 80-epoch training job. With the
# defaults below (9 timesteps x 1 block = 9 combinations), that's 9
# full training runs. Trim TIMESTEPS, or set FOLDS to a single
# fast fold first (e.g. FOLDS=(2)), before running the full grid.
#
# Usage:
#   chmod +x run_vde_sweep.sh
#   ./run_vde_sweep.sh
#
# Each combination's full stdout/stderr goes to its own log file under
# LOG_DIR, named by timestep/block, so a long-running sweep's output
# isn't lost if the terminal disconnects. One combination failing does
# NOT stop the rest of the sweep -- failures are recorded and reported
# in the summary at the end, same as this project's Python scripts
# skip-and-continue rather than abort a whole batch on one bad case.

set -uo pipefail  # fail fast on unset variables; do NOT use -e here --
                  # we want a failed combination to be recorded and the
                  # sweep to continue, not the whole script to exit.

# --- Edit these to control the sweep ---------------------------------

# Diffusion timestep values to sweep (VDEConfig.encoder_param["timestep"]).
if [[ -n "${VDE_TIMESTEPS:-}" ]]; then
    read -r -a TIMESTEPS <<< "$VDE_TIMESTEPS"
else
    TIMESTEPS=(0 50 100 150 200 250 500 750 900)
fi

# U-Net block values to sweep (VDEConfig.encoder_param["block_to_use"],
# valid range 0-17 per train_cv.py's --block_to_use help text).
if [[ -n "${VDE_BLOCKS:-}" ]]; then
    read -r -a BLOCKS <<< "$VDE_BLOCKS"
else
    BLOCKS=(15)
fi

# Which fold(s) to run for EVERY combination in the sweep. Leave empty
# ("") to use config.py's own train_folds (all 5, by default) for every
# combination -- the full, expensive option. Set to a single fold (e.g.
# "2") to sanity-check the whole grid quickly before committing to all 5
# folds for every combination.
# Which fold(s) to run for EVERY combination in the sweep. This is a
# bash ARRAY, not a plain string -- required so multiple folds pass
# through to train_cv.py's --folds (which takes multiple separate
# arguments, e.g. `--folds 1 2 3`) correctly, rather than as a single
# quoted "1 2 3" string that argparse would then fail to parse as ints.
#   FOLDS=()          -- empty (default): use config.py's own
#                         train_folds for every combination (all 5,
#                         typically) -- the full, expensive option.
#   FOLDS=(2)          -- a single fold, for a fast sanity check.
#   FOLDS=(1 2 3)       -- multiple specific folds, e.g. the screening
#                         subset recommended earlier in this project
#                         (folds 1, 2, 3 -- deliberately including fold
#                         2 as the most diagnostic fold, plus 1 and 3 as
#                         the historically most stable ones).
#   FOLDS=(all)         -- explicitly every fold (1..n_fold) -- same
#                         effect as leaving FOLDS=() empty, spelled out.
if [[ -n "${VDE_FOLDS:-}" ]]; then
    read -r -a FOLDS <<< "$VDE_FOLDS"
else
    FOLDS=(1 2 4)
fi

# Override config.py's epochs for this sweep. Leave empty ("") to use
# config.py's epochs as-is. Set a smaller number (e.g. "30") for a
# cheaper screening sweep -- especially useful paired with PATIENCE
# below, since early stopping means most combinations won't actually
# run this many epochs anyway; this just sets the ceiling.
EPOCHS="${VDE_EPOCHS:-60}"

# Early-stop each fold once the EMA combined score hasn't improved for this many
# consecutive epochs (see train_fold's docstring in train_cv.py). Leave
# empty ("") to disable early stopping and always train the full epoch
# budget. Recommended for a screening sweep: something like "15" or "20"
# -- long enough that a fold showing genuine, sustained improvement
# (e.g. fold 3 in past runs, which didn't peak until epoch ~71 of an
# 80-epoch budget) isn't cut off prematurely, but short enough to save
# real time on combinations/folds that plateau early.
PATIENCE="${VDE_PATIENCE:-15}"

# Optional checkpoint-family suffix. Use this whenever architecture or training
# settings change so a new sweep cannot overwrite an older experiment.
CHECKPOINT_TAG="${VDE_CHECKPOINT_TAG:-}"
if [[ -n "$CHECKPOINT_TAG" && ! "$CHECKPOINT_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid VDE_CHECKPOINT_TAG: use only letters, digits, '_' or '-'" >&2
    exit 2
fi
TAG_SUFFIX="${CHECKPOINT_TAG:+_$CHECKPOINT_TAG}"
VDE_PROFILE="${VDE_PROFILE:-}"
CLIP_LEN="${VDE_CLIP_LEN:-}"
STEP_SIZE="${VDE_STEP_SIZE:-}"
USE_DEMO="${VDE_USE_DEMO:-}"
CHECKPOINT_ROOT="${VDE_CHECKPOINT_ROOT:-}"

# Where to write per-combination logs.
LOG_DIR="${VDE_LOG_DIR:-./logs/vde_sweep_$(date +%Y%m%d_%H%M%S)}"

# -----------------------------------------------------------------------

mkdir -p "$LOG_DIR"
echo "Logs for this sweep: $LOG_DIR"
echo "Sweeping ${#TIMESTEPS[@]} timestep(s) x ${#BLOCKS[@]} block(s) = $((${#TIMESTEPS[@]} * ${#BLOCKS[@]})) combination(s)"
if [ "${#FOLDS[@]}" -gt 0 ]; then
    echo "Folds restricted to: ${FOLDS[*]} (not necessarily the full 5-fold run)"
else
    echo "Folds: using config.py's train_folds default for every combination"
fi
if [ -n "$EPOCHS" ]; then
    echo "Epochs: overridden to $EPOCHS for every combination"
else
    echo "Epochs: using config.py's epochs default for every combination"
fi
if [ -n "$PATIENCE" ]; then
    echo "Early stopping: patience=$PATIENCE epochs"
else
    echo "Early stopping: disabled (training the full epoch budget every time)"
fi
if [ -n "$CHECKPOINT_TAG" ]; then
    echo "Checkpoint tag: $CHECKPOINT_TAG"
fi
if [ -n "$VDE_PROFILE" ]; then
    echo "VDE profile: $VDE_PROFILE"
fi
if [ -n "$CLIP_LEN" ]; then
    echo "Clip length: $CLIP_LEN"
fi
if [ -n "$STEP_SIZE" ]; then
    echo "Step size: $STEP_SIZE"
fi
if [ "$USE_DEMO" = "1" ] || [ "$USE_DEMO" = "true" ]; then
    echo "Demographic fusion: enabled"
fi
echo ""

declare -a RESULTS=()  # each entry: "timestep,block,status,log_path"
n_total=$((${#TIMESTEPS[@]} * ${#BLOCKS[@]}))
n_done=0

for t in "${TIMESTEPS[@]}"; do
    for b in "${BLOCKS[@]}"; do
        n_done=$((n_done + 1))
        log_path="${LOG_DIR}/t${t}_block${b}.log"

        # Preserve a completed combination when re-running or extending a
        # sweep. A pooled file only counts as complete when it contains all
        # folds requested by this invocation.
        if [[ -n "$CHECKPOINT_ROOT" ]]; then
            pooled_path="${CHECKPOINT_ROOT}/block${b}/pooled_oof.csv"
        else
            pooled_path="./checkpoints/vde_cv_t${t}_block${b}${TAG_SUFFIX}/pooled_oof.csv"
        fi
        if [[ -f "$pooled_path" ]] && python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); c="fold" if "fold" in d else "fold_id"; assert sorted(map(int,d[c].unique())) == sorted(map(int,sys.argv[2:]))' "$pooled_path" "${FOLDS[@]}"; then
            echo "[$n_done/$n_total] timestep=$t block_to_use=$b -> SKIPPED (complete pooled OOF exists)"
            RESULTS+=("${t},${b},skipped_existing,${pooled_path}")
            echo ""
            continue
        fi

        cmd=(python train_cv.py --timestep "$t" --block_to_use "$b")
        if [ -n "$CHECKPOINT_TAG" ]; then
            cmd+=(--checkpoint_tag "$CHECKPOINT_TAG")
        fi
        if [ -n "$CHECKPOINT_ROOT" ]; then
            cmd+=(--checkpoint_root "$CHECKPOINT_ROOT")
        fi
        if [ -n "$VDE_PROFILE" ]; then
            cmd+=(--vde_profile "$VDE_PROFILE")
        fi
        if [ -n "$CLIP_LEN" ]; then
            cmd+=(--clip_len "$CLIP_LEN")
        fi
        if [ -n "$STEP_SIZE" ]; then
            cmd+=(--step_size "$STEP_SIZE")
        fi
        if [ "$USE_DEMO" = "1" ] || [ "$USE_DEMO" = "true" ]; then
            cmd+=(--use_demo)
        fi
        if [ "${#FOLDS[@]}" -gt 0 ]; then
            cmd+=(--folds "${FOLDS[@]}")
        fi
        if [ -n "$EPOCHS" ]; then
            cmd+=(--epochs "$EPOCHS")
        fi
        if [ -n "$PATIENCE" ]; then
            cmd+=(--patience "$PATIENCE")
        fi

        echo "[$n_done/$n_total] timestep=$t block_to_use=$b -> ${cmd[*]}"
        echo "    log: $log_path"

        start_ts=$(date +%s)
        if "${cmd[@]}" > "$log_path" 2>&1; then
            elapsed=$(( $(date +%s) - start_ts ))
            echo "    done in ${elapsed}s"
            RESULTS+=("${t},${b},ok,${log_path}")
        else
            elapsed=$(( $(date +%s) - start_ts ))
            echo "    FAILED after ${elapsed}s -- see $log_path"
            RESULTS+=("${t},${b},FAILED,${log_path}")
        fi
        echo ""
    done
done

# --- Summary -----------------------------------------------------------

summary_path="${LOG_DIR}/summary.csv"
echo "timestep,block_to_use,status,log_path" > "$summary_path"
for r in "${RESULTS[@]}"; do
    echo "$r" >> "$summary_path"
done

n_ok=0
n_failed=0
echo "=================== Sweep summary ==================="
printf "%-10s %-12s %-8s\n" "timestep" "block_to_use" "status"
for r in "${RESULTS[@]}"; do
    IFS=',' read -r t b status log_path <<< "$r"
    printf "%-10s %-12s %-8s\n" "$t" "$b" "$status"
    if [ "$status" = "ok" ] || [ "$status" = "skipped_existing" ]; then
        n_ok=$((n_ok + 1))
    else
        n_failed=$((n_failed + 1))
    fi
done
echo "======================================================="
echo "$n_ok / $n_total combinations succeeded, $n_failed failed"
echo "Full summary written to: $summary_path"
if [ "$n_failed" -gt 0 ]; then
    echo "Check the individual log files above for failed combinations before re-running just those."
    exit 1
fi
