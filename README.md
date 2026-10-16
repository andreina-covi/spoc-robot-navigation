# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.

**For agents continuing data-collection / cm-benchmark work**, read:

→ **[docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)** — design of `Collector`, RoomVisit placement trials, output schemas, performance notes, and QA usage.

→ **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)** — how `visible-pixels`, `bbox-area`, `min-side`, and `occupancy-ratio` are calculated.

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

Then set dataset / output paths:

```bash
. configure_variables.sh
```

This sets (edit paths in `configure_variables.sh` if needed):

- `OBJAVERSE_DATA_DIR`
- `OBJAVERSE_HOUSES_DIR`
- `OBJAVERSE_NAVIGATION_PATH` — where episode CSVs / images are written

---



## 2. Collect RoomVisit episodes (survey logs)

`RoomVisitTask` uses `Collector` during navigation. While the agent explores:

1. A pickupable is tracked the first time it has nav-camera mask pixels. Masks render on every agent step.
2. On episode end (`done` or `max_steps`), logs are saved under `$OBJAVERSE_NAVIGATION_PATH/<timestamp>/` with `images/` and `annotations/`.

Placement trials run in that same command, after the agent finishes. The house indexes are the ones already on the eval samples; they are passed in memory. Pass `--floor_radius_m`. The command writes `$OBJAVERSE_NAVIGATION_PATH/placement_trials/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`. Each pickupable is tried on floor points within that radius of landmarks in its room and on the 5 largest non-openable receptacles in that room. Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor, including Objaverse instances. One spawn per anchor. The object is restored only when the place succeeds. A failed restore reloads the house before the next pickupable. Each CSV row includes `radius_m`.

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

Increase `--eval_set_size` for more episodes. Details, schemas, and QA recipes: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md).

### Output folder layout

```text
$OBJAVERSE_NAVIGATION_PATH/
  placement_trials/
    placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
  <MM_DD_YYYY_HH_MM_SS_ffffff>/
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
```

Scene tag is `house_<index>` (not `Procedural`). Episode identity is the **folder** + `annotations/episode_meta` (not repeated on every CSV row).

**SPOC trajectory vs THOR navigability:** `navigation-*.csv` is the exact agent rollout (actions, success/fail, poses). `nav_graph-*.json` is environment reachability from `GetReachablePositions` (nodes + grid-adjacency edges), snapshotted at episode start and end. Do **not** treat the SPOC path as optimal.

Visibility columns on each navigation object row (`visible-pixels`, `bbox-area`, `min-side`, `occupancy-ratio`): see **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)**.

Summarize one export:

```bash
python scripts/summarize_episode_export.py --run_dir "$RUN"
```

---



## 3. Review collected data

```bash
ls -lt "$OBJAVERSE_NAVIGATION_PATH" | head
export RUN="$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
export SCENE=house_XXXXXX
cat "$RUN/annotations/episode_meta-${SCENE}.json"
```

Placement trials from that same command are `placement_trials/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`. The house indexes come from the eval samples.
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
| `configure_variables.sh`              | Dataset / navigation paths                                |


