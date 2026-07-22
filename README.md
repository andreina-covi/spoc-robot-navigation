# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.

**For agents continuing data-collection / cm-benchmark work**, read:

→ **[docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)** — design of `Collector`, RoomVisit invisible displacement, output schemas, realism rules, performance notes, QA usage, and known fixes.

→ **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)** — how `visible-pixels`, `bbox-area`, `min-side`, and `occupancy-ratio` are calculated.

---

## 1. Environment setup

```bash
. configure_variables.sh
```

This sets (edit paths in `configure_variables.sh` if needed):

- `OBJAVERSE_DATA_DIR`
- `OBJAVERSE_HOUSES_DIR`
- `OBJAVERSE_NAVIGATION_PATH` — where episode CSVs / images are written

---

## 2. Collect RoomVisit episodes (with invisible displacement + survey logs)

`RoomVisitTask` uses `Collector` during navigation. While the agent explores:

1. Pickupable objects seen in the **nav camera** are tracked.
2. After they leave the nav FOV for ≥2 steps, they may be relocated **in the same room** via `PlaceObjectAtPoint`, **only if the new pose stays out of the nav camera** (no pop-in).
3. On episode end (`done` or `max_steps`), logs are saved under `$OBJAVERSE_NAVIGATION_PATH/<timestamp>/` with `images/` and `annotations/`.

Run a short eval (1 episode):

```bash
. configure_variables.sh

python -m training.offline.online_eval --shuffle --eval_subset minival \
  --output_basedir /home/andreina/Documents/Programs/Dataset/logs \
  --test_augmentation --task_type RoomVisit \
  --eval_set_size 1 \
  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
  --house_set objaverse --wandb_logging False --num_workers 1 \
  --gpu_devices 0 \
  --training_run_id SigLIP-ViTb-3-CHORES-S \
  --local_checkpoint_dir /home/andreina/Documents/Programs/Dataset/checkpoints
```

Increase `--eval_set_size` for more episodes. Details, schemas, and QA recipes: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md).

### Output folder layout

```text
$OBJAVERSE_NAVIGATION_PATH/<MM_DD_YYYY_HH_MM_SS_ffffff>/
  images/                                 # RGB frames (img_<t>.png)
  annotations/                            # all CSV + JSON for the episode
    navigation-house_XXXXXX.csv
    objects-house_XXXXXX.csv
    doors-house_XXXXXX.csv
    object_state-house_XXXXXX.csv
    displacement_events-house_XXXXXX.csv
    displacement_debug-house_XXXXXX.csv
    passage_state-house_XXXXXX.csv
    region_trajectory-house_XXXXXX.csv
    world_layout-house_XXXXXX.json
    episode_meta-house_XXXXXX.json        # episode_id, episode_kind, counts
```

Scene tag is `house_<index>` (not `Procedural`). Episode identity is the **folder** + `annotations/episode_meta` (not repeated on every CSV row).

Visibility columns on each navigation object row (`visible-pixels`, `bbox-area`, `min-side`, `occupancy-ratio`): see **[docs/VISIBILITY_METRICS.md](docs/VISIBILITY_METRICS.md)**.

---

## 3. Review collected data

```bash
ls -lt "$OBJAVERSE_NAVIGATION_PATH" | head
export RUN="$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
export SCENE=house_XXXXXX
cat "$RUN/annotations/episode_meta-${SCENE}.json"
```

Check `num_displacements`, then `annotations/displacement_events-*.csv` and the corresponding `object_state` track (`in_camera_fov` true → false → move while false).  
Full review snippets and question examples: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) §§7–8.

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

## 5. Generate QA (legacy spatial templates)

For **invisible displacement / survey** items, use `object_state`, `displacement_events`, and `world_layout` (see [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)).

---

## 6. Code entry points

| File | Role |
|------|------|
| `docs/DATA_COLLECTION.md` | Full design context for agents |
| `docs/VISIBILITY_METRICS.md` | Formulas for nav visibility CSV columns |
| `tasks/room_visit_task.py` | RoomVisit step loop, hidden relocation, layout |
| `collector.py` | CSV/JSON logging, visibility tracking |
| `environment/spoc_objects.py` | `SPOCObject.get()` fix |
| `configure_variables.sh` | Dataset / navigation paths |
