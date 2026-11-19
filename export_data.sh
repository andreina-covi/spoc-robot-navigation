#!/usr/bin/env bash
# Export RoomVisit walks + placement trials for every house of one split, in chunks.
#
#   ./export_data.sh                      # all houses of $SPLIT
#   START=0 END=3 ./export_data.sh        # a test on houses 0-2
#
# Paths come from configure_variables.sh (not in git; copy configure_variables.example.sh
# and fill it in): OBJAVERSE_*, EXPORT_OUT_ROOT, EXPORT_CKPT_DIR, EXPORT_PYTHON, and
# optionally EXPORT_GPU, EXPORT_NUM_WORKERS, EXPORT_MEMORY_MAX.
#
# Resumable: rerun the same command after a crash or a failed house. A house with a
# finished episode (episode_meta-*.json) is not walked again; one without placement
# trials gets only the trials; an unfinished folder is renamed to .partial-house_*.
# Keep rerunning until every chunk log says "0 to walk".
#
# Run settings (environment variables):
#   SPLIT       val (16000 houses). The houses folder has no test file, and train
#               (160000 houses) is not supported: every chunk would load all of them
#   START, END  house index range [START, END); END not set = every house of the split
#               (counted from $OBJAVERSE_HOUSES_DIR/$SPLIT.jsonl.gz)
#   CHUNK       houses per online_eval run
#   RUN_ID, FLOOR_RADIUS_M  checkpoint run id and placement-trial floor radius
# Sourcing it (". export_data.sh") would run it in your terminal's shell, and its
# exit / set -e would close the terminal: refuse and say how to run it.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run it as ./export_data.sh or bash export_data.sh, not with source or '.'" >&2
  return 1
fi
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [[ ! -f configure_variables.sh ]]; then
  echo "Missing configure_variables.sh: cp configure_variables.example.sh configure_variables.sh and set your paths" >&2
  exit 1
fi
. ./configure_variables.sh

for var in OBJAVERSE_DATA_DIR OBJAVERSE_HOUSES_DIR EXPORT_OUT_ROOT EXPORT_CKPT_DIR EXPORT_PYTHON; do
  if [[ -z "${!var:-}" || "${!var}" == /path/to/* ]]; then
    echo "Set $var in configure_variables.sh" >&2
    exit 1
  fi
done

SPLIT=${SPLIT:-val}
CHUNK=${CHUNK:-3}
START=${START:-0}
RUN_ID=${RUN_ID:-SigLIP-ViTb-3-CHORES-S}
FLOOR_RADIUS_M=${FLOOR_RADIUS_M:-1.2}
OUT_ROOT=$EXPORT_OUT_ROOT
CKPT_DIR=$EXPORT_CKPT_DIR
PYTHON=$EXPORT_PYTHON
GPU=${EXPORT_GPU:-0}
NUM_WORKERS=${EXPORT_NUM_WORKERS:-4}
MEMORY_MAX=${EXPORT_MEMORY_MAX:-}

HOUSES_FILE="$OBJAVERSE_HOUSES_DIR/$SPLIT.jsonl.gz"
[[ -f "$HOUSES_FILE" ]] || { echo "No house file $HOUSES_FILE (use SPLIT=val)" >&2; exit 1; }
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
  # $GPU is unquoted on purpose: "0 1" becomes two ids.
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
  grep -h "^\[all_houses\]" "$log" | tail -n 1 | sed 's/^/  /' || true
  # online_eval's own run folder (local wandb metrics) is not episode data; the chunk log
  # above keeps everything it printed.
  rm -rf "$LOG_DIR/online_eval"
done

done_count=$(find "$OBJAVERSE_NAVIGATION_PATH" -path '*/annotations/episode_meta-*.json' | wc -l)
echo "finished episodes in $OBJAVERSE_NAVIGATION_PATH: $done_count; failed chunks this pass: $failed"
exit $((failed > 0))
