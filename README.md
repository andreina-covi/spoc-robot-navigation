# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.

**For agents continuing data-collection / cm-benchmark work**, read:

→ **[docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)** — design of `Collector`, RoomVisit placement trials, output schemas, performance notes, and QA usage.

→ **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)** — how `visible-pixels`, `bbox-area`, `min-side`, `occupancy-ratio`, and `mark-x` / `mark-y` are calculated.

---



## 1. Environment setup

Python **3.9+**. Install the stack from `[requirements.txt](requirements.txt)` (simulation, training/eval, data collection, and the custom Stretch AI2-THOR build from AllenAI’s index):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

That single `pip` command also installs `ai2thor==0+5d0ab8ab…` via `--extra-index-url https://ai2thor-pypi.allenai.org` (no second install step). `CLIP` and `nltk` come from pinned git commits, so `git` must be on `PATH`.

Leave these pins unless you re-test online eval: `torch==2.0.1`, `numpy==1.23.5`, `allenact==0.5.4`.

Then set dataset / output paths. `configure_variables.sh` is local to each machine and is not in git: copy the template and fill in your paths.

```bash
cp configure_variables.example.sh configure_variables.sh   # once per machine, then edit it
. configure_variables.sh
```

This sets:

- `OBJAVERSE_DATA_DIR`
- `OBJAVERSE_HOUSES_DIR`
- `OBJAVERSE_NAVIGATION_PATH` — where episode CSVs / images are written
- `EXPORT_OUT_ROOT`, `EXPORT_CKPT_DIR`, `EXPORT_PYTHON` (and optional `EXPORT_GPU`, `EXPORT_NUM_WORKERS`, `EXPORT_MEMORY_MAX`) — used by `export_data.sh` (see [Export every house of a split](#export-every-house-of-a-split-chunked-resumable))

---



## 2. Collect RoomVisit episodes (survey logs)

`RoomVisitTask` uses `Collector` during navigation. While the agent explores:

1. A pickupable is tracked the first time it has nav-camera mask pixels. Masks render on every agent step.
2. When a movable object is close but out of view, the agent glances at it: a fixed number of 45° turns toward the object, then back. Those frames are marked `scripted-glance` in `navigation-*.csv`.
3. On episode end (`done`, `max_steps` or `timeout`), logs are saved under `$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX/` with `images/` and `annotations/` (`house_XXXXXX_2/`, `_3`… when that house already has a folder, so a re-collection never overwrites one).

Movable objects are every pickupable and, with `placement.include_moveable`, every `moveable` object up to `placement.moveable_max_mass_kg` with nothing resting on it (chairs, plants, lamps, bins). The glance and the placement trials use this same set.

Placement trials run in that same command, after the agent finishes. The house indexes are the ones already on the eval samples; they are passed in memory. Pass `--floor_radius_m`. The command writes `$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX/annotations/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`, next to the other episode files. Each movable object is tried on floor points beside the landmarks in its room, within that radius, and on the 5 largest non-openable receptacles in that room. A non-pickupable object that stands on the floor, such as a chair, gets floor trials only. Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor, including Objaverse instances. Doors and doorways are excluded. A receptacle place is kept only when that receptacle is a parent of the object; up to 3 spawn points nearest the top of the receptacle are tried. After a successful place the object is teleported back to its exact original pose. Each CSV row includes `obj-kind` (`pickupable` or `moveable`), `radius_m` and `parent-receptacle`. A house that fails does not stop the run: the simulator is restarted and the next house is tried.

Every collection setting (episode timeout, policy-step cap, glance, placement) is in `configs/collection.yaml`. The file used for a run is copied into `episode_meta-*.json` as `collection_config`.

Run it:

```bash
. configure_variables.sh

python -m training.offline.online_eval --shuffle --eval_subset minival \
  --output_basedir /home/andreina/Documents/Programs/Dataset/logs \
  --test_augmentation --task_type RoomVisit \
  --eval_set_size 1 \
  --floor_radius_m 1.2 \
  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
  --house_set objaverse --wandb_logging False --num_workers 1 \
  --gpu_devices 0 \
  --training_run_id SigLIP-ViTb-3-CHORES-S \
  --local_checkpoint_dir /home/andreina/Documents/Programs/Dataset/checkpoints
```

Increase `--eval_set_size` for more episodes.

To walk every house of a split instead of the benchmark samples, add `--all_houses` with `--eval_subset val` or `--eval_subset test` (the Objaverse house set, `--house_set objaverse`, has no test split), and optionally `--house_start N --house_end M` (house indexes N to M-1). This gives one RoomVisit episode per house, with the house's split recorded. The start pose is taken from the house's reachable positions with a heading that is a multiple of 30°, seeded by the house index, so a rerun starts each house in the same place. `--eval_set_size` and `--shuffle` do not apply. `--eval_subset test` without `--all_houses` is an error, because the benchmark samples are val samples and would point at test houses with the same index. Placement trials run afterwards for the same houses.

### Export every house of a split (chunked, resumable)

`export_data.sh` runs the command above with `--all_houses` over a whole split, in chunks of `CHUNK` houses, so a crash costs one chunk and not the whole run.

```bash
cd spoc-robot-training                      # any directory works; the script finds its own
SPLIT=val START=0 END=3 ./export_data.sh    # test on houses 0, 1 and 2
SPLIT=val CHUNK=500 ./export_data.sh        # every val house (END not set = all of them)
```

Run it as `./export_data.sh` or `bash export_data.sh`. Never run it with `source` or `.`: it refuses, because in your own shell its `exit` would close the terminal. On a laptop, `systemd-inhibit ./export_data.sh` keeps the machine from sleeping during the run.

Settings:

| Variable | Where | Meaning |
|---|---|---|
| `SPLIT` | command line | `val` (default). The Objaverse house folder has `val.jsonl.gz` (16,000 houses) and `train.jsonl.gz` (160,000 houses), and no test file. `train` is not supported: every chunk would load all 160,000 houses into memory. |
| `START` | command line | First house index (default 0). |
| `END` | command line | House index to stop before. **Not set = the number of houses in `$OBJAVERSE_HOUSES_DIR/$SPLIT.jsonl.gz`**, i.e. the whole split. |
| `CHUNK` | command line | Houses per `online_eval` run (one model load each). |
| `RUN_ID`, `FLOOR_RADIUS_M` | command line | Checkpoint run id (default `SigLIP-ViTb-3-CHORES-S`) and placement radius (default 1.2 m). |
| `EXPORT_OUT_ROOT` | `configure_variables.sh` | Output root: episodes in `$EXPORT_OUT_ROOT/<split>/nav/house_XXXXXX/`, logs in `$EXPORT_OUT_ROOT/<split>/logs/chunk_<start>_<end>.log`. It replaces `OBJAVERSE_NAVIGATION_PATH` for this script. |
| `EXPORT_CKPT_DIR`, `EXPORT_PYTHON` | `configure_variables.sh` | Checkpoint folder and the Python of the spoc environment. |
| `EXPORT_GPU`, `EXPORT_NUM_WORKERS` | `configure_variables.sh` | GPU ids, space separated (default `0`), and workers per chunk (default 4). |
| `EXPORT_MEMORY_MAX` | `configure_variables.sh` | e.g. `10G`: each chunk runs under `systemd-run --user --scope -p MemoryMax=...`, so running out of memory kills that chunk, not the desktop. Empty = no cap. A 15 GB laptop needs 1 worker and `10G`. |

The script stops with a message when `configure_variables.sh` is missing, when a required path is still a `/path/to/...` placeholder, or when the house file or checkpoint folder does not exist.

**Resuming.** Rerun the same command after a crash or a failed house; repeat until every chunk log says `0 to walk`. Each chunk log starts with a line like `[all_houses] val houses 0-500: 12 to walk, 488 already walked (3 of them still need placement trials); 1 unfinished folder(s) renamed to .partial-*`:

- a house whose folder has `annotations/episode_meta-*.json` (written last, when the episode ends) is finished and is not walked again;
- a finished house without `placement_candidates-*.csv` gets only the placement trials;
- an unfinished folder of a house about to be walked again is renamed to `.partial-house_XXXXXX` (kept, but cm-benchmark skips dot folders), so the new walk gets the plain `house_XXXXXX` name;
- a house whose walk fails is reported (`no finished episode for val houses [...]; skipped`) and the rest of the chunk still gets its placement trials.

A worker that hits an error stops, and the other workers take its houses; if every worker of a chunk stops, the rerun walks what is left. Each exported house takes about 45-80 MB, so the full val split needs roughly 750 GB to 1.2 TB.

Details, schemas, and QA recipes: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md).

### Output folder layout

```text
$OBJAVERSE_NAVIGATION_PATH/
  house_XXXXXX/                             # one episode, named after the house
    images/                                 # RGB frames (img_<t>.png)
    annotations/                            # all CSV + JSON for the episode
      navigation-house_XXXXXX.csv
      objects-house_XXXXXX.csv
      doors-house_XXXXXX.csv
      object_state-house_XXXXXX.csv
      passage_state-house_XXXXXX.csv
      world_layout-house_XXXXXX.json
      nav_graph-house_XXXXXX.json           # AI2-THOR reachability (≠ SPOC path)
      episode_meta-house_XXXXXX.json        # episode_id, episode_kind, counts
      placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
```

A second episode of the same house goes to `house_XXXXXX_2/` (then `_3`, ...), so nothing is overwritten. Scene tag is `house_<index>` (not `Procedural`). Episode identity is the **folder** + `annotations/episode_meta` (not repeated on every CSV row).

**SPOC trajectory vs THOR navigability:** `navigation-*.csv` is the exact agent rollout (actions, success/fail, poses). `nav_graph-*.json` is environment reachability from `GetReachablePositions` (nodes + grid-adjacency edges), snapshotted at episode start and end. Do **not** treat the SPOC path as optimal.

Visibility columns on each navigation object row (`visible-pixels`, `bbox-area`, `min-side`, `occupancy-ratio`, `mark-x`, `mark-y`): see **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)**. Agent room is `current-room` / `current-room-type`. Object room is `obj-room` / `obj-room-type` on the same row (and `room-id` / `room-type` in `objects-*.csv`). `objects-*.csv` also has the THOR properties `pickupable`, `moveable` and `mass`.

Summarize one export:

```bash
python scripts/summarize_episode_export.py --run_dir "$RUN"
```

---



## 3. Review collected data

```bash
ls -lt "$OBJAVERSE_NAVIGATION_PATH" | head
export RUN="$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX"
export SCENE=house_XXXXXX
cat "$RUN/annotations/episode_meta-${SCENE}.json"
```

Placement trials from that same command are `$RUN/annotations/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`. The house indexes come from the eval samples.
`success` marks a place that was read and then restored.
Full review snippets: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) §§7–8.

---

## 4. Post-process spatial descriptions (existing pipeline)

```bash
python -m spatial_data_generation \
  --csv_path_navigation "$RUN/annotations/navigation-${SCENE}.csv" \
  --csv_path_objects "$RUN/annotations/objects-${SCENE}.csv" \
  --json_path_dict "$RUN/annotations/jsons" \
  --json_filename structured_data_angle.json
```

Example (legacy flat layout / `Procedural` name; new runs use `annotations/` + `house_XXXXXX`):

```bash
python -m spatial_data_generation \
  --csv_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/05_06_2026_17_02_54_768901/navigation-Procedural.csv \
  --csv_path_objects /home/andreina/Documents/Programs/Dataset/Generated/navigation/05_06_2026_17_02_54_768901/objects-Procedural.csv \
  --json_path_dict /home/andreina/Documents/Programs/Dataset/Generated/navigation/05_06_2026_17_02_54_768901/jsons \
  --json_filename structured_data_angle.json
```

---

## 5. Normalize object types (after export, before cm-benchmark)

New exports write descriptive `obj-type`, `obj-synset`, and `assetId` while collecting. This script does the same fill for older episodes whose objects table still says `Undefined`, and prints confusable pairs from the synsets.

`scripts/normalize_object_types.py` only edits files already written by the collector. It does not start AI2-THOR. The path can be one episode folder, its `annotations/` directory, or any parent of episode folders. Run it after an episode export and before building cm-benchmark items.

```bash
python scripts/normalize_object_types.py --apply "$OBJAVERSE_NAVIGATION_PATH" --split val
```

Synset sources:

- THOR types use `AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET` (`Fridge` and an Objaverse refrigerator can share `refrigerator.n.01`).
- Objaverse assets use `annotation["synset"]` for the house `assetId`.
- If the house or the asset is missing, the name after `Obja` in the obj-id is the first wordnet2022 noun synset and the report source is `id_name`.

`--split` is required for annotation lookup when `episode_meta` has a house index but no split. The script does not guess the split. The first apply copies each changed file to `<name>.orig.<ext>` and later runs read that copy. After writing, the script prints every object pair in the episode that has the same synset, a hypernym link in either direction (`is_hypernym_of`), or a shared direct hypernym.

## 6. Generate QA (legacy spatial templates)

For **placement-trial / survey** items, use `placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`
from `scripts/placement_trials.py` and `world_layout` (see [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)).

---

## 7. Code entry points


| File                                  | Role                                                      |
| ------------------------------------- | --------------------------------------------------------- |
| `docs/DATA_COLLECTION.md`             | Full design context for agents                            |
| `docs/VISIBILITY_METRICS.md`          | Formulas for nav visibility CSV columns                   |
| `tasks/room_visit_task.py`            | RoomVisit step loop, layout, nav graph                    |
| `scripts/placement_trials.py`         | Placement trials after RoomVisit navigation in the same command |
| `collector.py`                        | CSV/JSON logging, visibility tracking, `nav_graph` export |
| `utils/nav_graph_export.py`           | Reachable-position nodes + grid-adjacency edges           |
| `scripts/summarize_episode_export.py` | One-episode nav/trajectory summary                        |
| `scripts/normalize_object_types.py`   | Post-export obj-type / obj-synset (no simulator)          |
| `environment/spoc_objects.py`         | `SPOCObject.get()` fix                                    |
| `requirements.txt`                    | Python deps (including Stretch AI2-THOR)                  |
| `configure_variables.example.sh`      | Template for the local `configure_variables.sh` (paths)   |
| `export_data.sh`                      | Chunked, resumable export of every house of a split       |


