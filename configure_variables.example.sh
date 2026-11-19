#!/bin/bash
# Template for the machine-specific paths. Copy it and fill in your paths:
#
#   cp configure_variables.example.sh configure_variables.sh
#
# configure_variables.sh is in .gitignore, so each machine keeps its own copy.

# Objaverse assets / annotations
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets

# Houses (train.jsonl.gz, val.jsonl.gz)
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_assets/objaverse_houses/houses_2023_07_28

# Where single online_eval runs write episode folders (export_data.sh sets its own per split)
export OBJAVERSE_NAVIGATION_PATH=/path/to/Generated/nav

# --- export_data.sh ---------------------------------------------------------
# Output root: walks in $EXPORT_OUT_ROOT/<split>/nav, logs in $EXPORT_OUT_ROOT/<split>/logs
export EXPORT_OUT_ROOT=/path/to/Generated/export
# Folder with the SPOC checkpoint (--local_checkpoint_dir)
export EXPORT_CKPT_DIR=/path/to/checkpoints
# Python of the spoc environment
export EXPORT_PYTHON=python
# Optional, defaults shown:
# export EXPORT_GPU="0"            # GPU ids, space separated ("0 1")
# export EXPORT_NUM_WORKERS=4
# export EXPORT_MEMORY_MAX=        # e.g. 40G: cap each chunk with systemd-run (empty = no cap)
