#!/usr/bin/env bash
# Export RoomVisit walks + placement trials for every house of one split, in chunks.
#
#   ./export_data.sh                      # all houses of $SPLIT
#   START=0 END=3 ./export_data.sh        # a test on houses 0-2
#
# Resumable: rerun the same command after a crash or a failed house. A house with a
# finished episode (episode_meta-*.json) is not walked again; one without placement
# trials gets only the trials; an unfinished folder is renamed to .partial-house_*.
# Keep rerunning until every chunk log ends with "0 to walk".
#
# Settings (environment variables; the defaults are for this machine, override on the server):
#   SPLIT       val (16000 houses). $OBJAVERSE_HOUSES_DIR has no test file, and train
#               (160000 houses) is not supported: every chunk would load all of them
#   START, END  house index range [START, END); END defaults to the split's house count
#   CHUNK       houses per online_eval run
#   OUT_ROOT    output root: walks in $OUT_ROOT/$SPLIT/nav, eval logs in $OUT_ROOT/$SPLIT/logs
#   GPU         GPU ids, space separated ("0 1")
#   PYTHON, NUM_WORKERS, CKPT_DIR, RUN_ID, FLOOR_RADIUS_M
#   MEMORY_MAX  e.g. 40G: run each chunk under systemd-run with that memory cap (empty = no cap)
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
. ./configure_variables.sh

SPLIT=${SPLIT:-val}
CHUNK=${CHUNK:-3}
START=${START:-0}
OUT_ROOT=${OUT_ROOT:-/home/andreina/Documents/Programs/Dataset/Generated/export}
PYTHON=${PYTHON:-python}
NUM_WORKERS=${NUM_WORKERS:-4}
GPU=${GPU:-0}
CKPT_DIR=${CKPT_DIR:-/home/andreina/Documents/Programs/Dataset/checkpoints}
RUN_ID=${RUN_ID:-SigLIP-ViTb-3-CHORES-S}
FLOOR_RADIUS_M=${FLOOR_RADIUS_M:-1.2}
MEMORY_MAX=${MEMORY_MAX:-}

HOUSES_FILE="$OBJAVERSE_HOUSES_DIR/$SPLIT.jsonl.gz"
[[ -f "$HOUSES_FILE" ]] || { echo "No house file $HOUSES_FILE (SPLIT=val; this folder has no test file)" >&2; exit 1; }
if [[ -z "${END:-}" ]]; then
  echo "Counting houses in $HOUSES_FILE ..."
  END=$(zcat "$HOUSES_FILE" | wc -l)
fi
[[ -d "$CKPT_DIR" ]] || { echo "No checkpoint dir $CKPT_DIR" >&2; exit 1; }

# The collector writes under $OBJAVERSE_NAVIGATION_PATH; one folder per split.
export OBJAVERSE_NAVIGATION_PATH="$OUT_ROOT/$SPLIT/nav"
LOG_DIR="$OUT_ROOT/$SPLIT/logs"
mkdir -p "$OBJAVERSE_NAVIGATION_PATH" "$LOG_DIR"

RUNNER=()
if [[ -n "$MEMORY_MAX" ]]; then
  RUNNER=(systemd-run --user --scope -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 --quiet)
fi

echo "split=$SPLIT houses $START-$END chunk=$CHUNK -> $OBJAVERSE_NAVIGATION_PATH"
failed=0
for ((s = START; s < END; s += CHUNK)); do
  e=$((s + CHUNK)); ((e > END)) && e=$END
  log="$LOG_DIR/chunk_${s}_${e}.log"
  echo "$(date '+%F %T') houses $s-$e (log $log)"
  if ! "${RUNNER[@]}" "$PYTHON" -m training.offline.online_eval \
      --eval_subset "$SPLIT" --all_houses --house_start "$s" --house_end "$e" \
      --output_basedir "$LOG_DIR/online_eval" \
      --test_augmentation --task_type RoomVisit --floor_radius_m "$FLOOR_RADIUS_M" \
      --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
      --house_set objaverse --wandb_logging False --num_workers "$NUM_WORKERS" --gpu_devices $GPU \
      --total_num_videos 0 \
      --training_run_id "$RUN_ID" --local_checkpoint_dir "$CKPT_DIR" \
      >>"$log" 2>&1; then
    echo "  chunk $s-$e failed (see $log); rerun the script to resume it"
    failed=$((failed + 1))
  fi
  grep -h "^\[all_houses\]" "$log" | tail -n 2 | sed 's/^/  /' || true
done

done_count=$(find "$OBJAVERSE_NAVIGATION_PATH" -path '*/annotations/episode_meta-*.json' | wc -l)
echo "finished episodes in $OBJAVERSE_NAVIGATION_PATH: $done_count; failed chunks this pass: $failed"
exit $((failed > 0))
