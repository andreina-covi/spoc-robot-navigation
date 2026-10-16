# Data collection for cm-benchmark (RoomVisit + invisible displacement)

> **Audience:** future agents / developers continuing this work.  
> **Status:** implemented and exercised on RoomVisit online eval.  
> **Primary code:** `collector.py`, `tasks/room_visit_task.py`, `environment/spoc_objects.py`

This document summarizes **why** collection was extended, **how** it works, **what files** are written, known pitfalls, and how logs support QA for `invisible_displacement` / survey constructs.

---

## 1. Goal

Extend SPOC RoomVisit navigation logging so post-processing / **cm-benchmark** can build items for:

| Construct | Need |
|-----------|------|
| `invisible_displacement` | Per house, each pickupable is trial-placed on floor points within the given radius of landmarks in its room and on the 5 largest non-openable receptacles in that room. Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor, including Objaverse instances. One spawn per anchor. Restored only when the place succeeds. A failed restore reloads the house before the next pickupable. Rows are `placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv` and include `radius_m`. |
| `survey_knowledge` | Multi-room trajectory + layout connectivity / door state |

Facing / intrinsic “front of object” is **out of scope** for this pass.

Episode kind tagged in meta: `invisible_displacement`. One RoomVisit command navigates, records the house indexes, then trial-places objects in those houses. Navigation itself does not move objects.

---

## 2. Architecture

```text
online_eval (RoomVisit)
  └─ for each house in the eval samples
       RoomVisitTask: agent navigates, Collector writes the episode
  └─ after the agent finishes
       pass those sample house indexes to placement trials
       write placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
```

| Module | Role |
|--------|------|
| `tasks/room_visit_task.py` | World layout, door sampling, pickupable tracking |
| `scripts/placement_trials.py` | Placement trials for the houses just navigated. Called at the end of the RoomVisit command. |
| `collector.py` | Visibility tracking, CSV/JSON writers, episode folder under `OBJAVERSE_NAVIGATION_PATH` |
| `environment/spoc_objects.py` | `SPOCObject.get()` fixed so `.get("pickupable", False)` works |

Scene naming: ProcTHOR `sceneName` is always `"Procedural"`. Files use `house_<zero-padded house_index>` from `task_info["house_index"]`.

---

## 3. Output layout

Path: `$OBJAVERSE_NAVIGATION_PATH/`  
(`configure_variables.sh` sets `OBJAVERSE_NAVIGATION_PATH`).

```text
$OBJAVERSE_NAVIGATION_PATH/
  placement_trials/
    placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
  <timestamp>/                            # one episode
    images/
    annotations/
      navigation-house_XXXXXX.csv
      ...
```

Each episode root has **two** sibling folders:

```text
<timestamp>/
  images/                                 # RGB frames
    img_<t>.png
  annotations/                            # structured CSV + JSON for post-processing
    navigation-house_XXXXXX.csv
    objects-house_XXXXXX.csv
    doors-house_XXXXXX.csv
    object_state-house_XXXXXX.csv
    passage_state-house_XXXXXX.csv
    world_layout-house_XXXXXX.json
    nav_graph-house_XXXXXX.json
    episode_meta-house_XXXXXX.json
```

**Do not** repeat `episode_id` / `scene_id` / `episode_kind` on every CSV row. Those live in `annotations/episode_meta-*.json` and the folder name. Consumers should join on folder + `timestep` / `obj-id` / `node_id`. Navigation `path` columns still point at files under `images/`.

`episode_meta-*.json` also stores run geometry and agent constants needed for offline recomputation:

| Section | Fields |
|---------|--------|
| `camera` | `width`, `height`, `frame_size_px`, `fov_vertical_deg` (nav / INTEL) |
| `agent` | `movement_constant`, `rotation_deg`, `horizon_deg`, `arm_move_constant`, `wrist_rotation_deg`, `reachability_grid_size` |
| `house_index`, `split` | House line in `OBJAVERSE_HOUSES_DIR/<split>.jsonl.gz`. Split is the one the evaluator loaded. |
| `git_commit`, `config` | Collector commit (`-dirty` when the tree has uncommitted edits) and the motion, stride, and horizon settings for this run |
| `motion` | Move distances and rotation angles the agent used, and whether `nav_graph` `grid_size`, `agent_move_m`, and `agent_rotation_deg` match them. A mismatch is also printed at save. |
| `nav_graph` | pointer + node/edge counts for start/end snapshots |
| `visibility_filters` | export policy note (no hard thresholds at collection) |

### Trajectory vs navigability (important)

| Source | What it is | What it is not |
|--------|------------|----------------|
| `navigation-*.csv` | Exact SPOC rollout: ordered steps, `ag-action`, `action_success`, agent/camera pose, FOV object rows | Not an optimal path; may include fails, rotations, backtracks |
| `nav_graph-*.json` | AI2-THOR `GetReachablePositions` standable cells + **exported** grid-adjacency edges | Not the agent path; not Euclidean k-NN between nearby points |
| `world_layout-*.json` | Coarse room/door survey graph | Not fine agent navigation |

**`nav_graph-*.json` structure:**

- Top-level: `episode_id`, `scene_id`, `snapshots`, `notes`
- `snapshots.episode_start` / `snapshots.episode_end`: each has `nodes`, `edges`, `params`, `door_states_at_snapshot`, `coordinate_frame`
- Nodes: `node_id`, `x`, `y`, `z` (AI2-THOR meters; **y up**, motion in **xz**)
- Edges: undirected pairs (`from_node`, `to_node`, `distance_xz`, `cost`, `bidirectional=true`) for **8-connected** neighbors on the reachability grid (`grid_size = 0.75 * agent_move_m`, default 0.15 m). Reconstruction rule is also recorded in `params.edge_rule` so consumers can rebuild the same edges from `nodes` alone if desired.
- `params`: `grid_size`, `agent_move_m` (0.2), `agent_rotation_deg` (45), `edge_connectivity`, `thor_action`, `snap_to_grid=false`

**Scene state:** start snapshot is taken at task init (before agent steps). End snapshot reuses that same door state. `doors` and `passage_state` record each door once, including `is_open`. RoomVisit does not open or close doors. Prefer the start nav graph for the map at rollout begin.

**Assumptions / limitations:**

- Edges are **grid adjacency on the THOR reachability sample**, not proven single-action success for every SPOC heading (agent move is 0.2 m; sample spacing is 0.15 m). Use trajectory `action_success` for what actually happened.
- Two nearby Euclidean points are **not** connected unless they are adjacent on that grid.
- We do **not** export a heuristic of which nav edges the agent “knows”; only primitive FOV/visibility in `navigation` / `object_state`.
- Reachability is fixed for the episode. RoomVisit does not open or close doors, so the end nav graph uses the same door state as the start.

Summarize one run:

```bash
python scripts/summarize_episode_export.py --run_dir "$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
```

### Which objects go where

| Output | Objects included |
|--------|------------------|
| `navigation-*.csv` | Non-structural FOV objects with `visible-pixels > 0`, including numeric ids such as `6|5`. Each step has `ag-action`, `action_success` (`lastActionSuccess`), and the agent pose after that action. |
| `objects-*.csv` | Catalog of those FOV objects. `obj-id` is unchanged. `obj-type` and `obj-synset` are filled at export (`describe_object`). Also `assetId`, `room-id`, and the axis-aligned box `aabb-min-*` / `aabb-max-*`. |
| `object_state-*.csv` | **Pickupable** objects tracked after nav mask pixels at least once |
| `current-room` (in `navigation-*.csv`) | **Agent–room** membership (do **not** treat Floor/Wall as the room object) |

**Export vs filter (recommended policy):**

| Stage | What is dropped at collection time | Who decides the rest |
|-------|------------------------------------|----------------------|
| Navigation / objects CSV | Structural (Wall/Floor/Ceiling/Room) + zero mask pixels. Numeric ids are kept. | Post-processing via metrics |

**Why export-all-with-metrics for nav:** thresholds can be retuned without re-running THOR; different tasks can use different cutoffs.

**Visibility metrics on each navigation object row** (formulas and caveats:
**[VISIBILITY_METRICS.md](VISIBILITY_METRICS.md)**):

| Column | Meaning |
|--------|---------|
| `expected-bbox-area` | On-screen projected area from 3D OBB/AABB corners (`geometry.py`; clamped to frame) |
| `expected_cmin` / `expected_cmax` / `expected_rmin` / `expected_rmax` | Pixel bounds of that expected bbox (`None` if not projectable) |
| `ang-width-deg` / `ang-height-deg` | Angular size of that expected bbox (degrees) |
| `visible-pixels` | Mask pixels of the object inside its detection bbox |
| `bbox-area` | Detection box area `(cmax-cmin)*(rmax-rmin)` |
| `min-side` | `min(width, height)` of the detection bbox |
| `occupancy-ratio` | `visible-pixels / bbox-area` |

Example post-process keep rule: apply your own cutoffs on
`visible-pixels` / `min-side` / `occupancy-ratio`.

Fully hidden / behind other geometry never appear in detections. Out-of-camera objects are
not in `navigation` for that step. Hidden pickupables appear only in `object_state`
(after they were tracked via mask pixels at least once).

**Spatial relations (post-processing):**
- Agent ↔ object: use `navigation` rows + agent pose vs object pose / bbox.
- Agent ↔ room: use `current-room` / `region_trajectory` (and `world_layout`), not structural mesh rows.
- Agent ↔ navigable map: snap poses to `nav_graph` nodes; do **not** replace the trajectory with a shortest path on that graph.

### Important columns

**`object_state-*.csv`** (per timestep × tracked pickupable; includes hidden rows):

- `timestep`, `obj-id`, `obj-type`, pose
- `visible` — THOR metadata (`visibilityDistance` LOS)
- `in_camera_fov` — nav-image presence when a mask was rendered.
- `parent_receptacle`, `parent_receptacles`, `is_inside_receptacle`, `receptacle_is_open`

Tracking starts when a **pickupable** first has mask pixels in the agent camera.
HousePlant / Fridge / counters / windows stay in `navigation` only.

**`placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`** is written by `scripts/placement_trials.py`, not by the navigation episode. `anchor-id` is the receptacle id or the landmark id. `anchor-kind` is `receptacle` or `floor`. `pos-*` is the resolved pose when `success` is true. A successful place is restored before the next anchor. Episodes collected before this change may still contain `displacement_events`, `displacement_candidates`, and `displacement_debug`; those files are not written now.

**`navigation-*.csv`:** one row group per step with `ag-action`, `action_success` (`lastActionSuccess`), and `ag-pos-*` / `ag-rot-*` after the action. Object rows are added only when `visible-pixels > 0`. Numeric ids are included. Hidden tracked pickupables stay in `object_state`.

**`objects-*.csv`:** one row per unique exported object. `obj-id` is the THOR id, including values like `6|5`. `obj-type` and `obj-synset` come from `describe_object`. `aabb-min-*` / `aabb-max-*` are the 3D axis-aligned corners. `room-id` is the room that contains the object.

**`world_layout-*.json`:** each region has `polygon` (floor outline) and `area` (square meters on xz). `doors` and `passage_state` record each door once at the start, including whether it is open.

**`nav_graph-*.json`:** fine-grained THOR reachability (see Trajectory vs navigability above).

---

## 4. Placement trials

The RoomVisit command already knows the house indexes: they are the `house_id` values on the eval samples. After the agent finishes, that same command passes those indexes to the placement trials. Nothing is written just to remember the indexes.

The CSV is `$OBJAVERSE_NAVIGATION_PATH/placement_trials/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`. The radius is `--floor_radius_m` on the RoomVisit command.

For each pickupable:

1. Floor: the closest floor spawn within that radius of each landmark in its room. Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor height, including Objaverse instances. Doors and doorways are excluded. A room with no floor uses the lowest floor height in the house. `--top_landmarks` changes that count. One point per landmark. Object poses are read after 10 physics steps.
2. Receptacles: the 5 largest non-openable receptacles in that room (`receptacle_salience` is the axis-aligned box volume). Floor, wall, ceiling, room, and openable receptacles are excluded.
3. Up to 3 spawn coordinates per receptacle, the ones closest to the center of its top face. Spawn lists are cached per receptacle. One floor spawn per landmark.
4. `PlaceObjectAtPoint` with `forceKinematic`. A receptacle row is `success` only when that receptacle is in `parentReceptacles` after the place. `parent-receptacle` records those ids. The original pose is restored only when the place action succeeds. If the restore fails, the house is reloaded before the next pickupable.
5. Columns: `obj-id`, `anchor-id`, `anchor-kind` (`receptacle` or `floor`), `pos-*`, `success`, `radius_m`, `parent-receptacle`. The split and radius are also in the file name.

---

## 5. Performance constraints (do not regress)

Earlier freezes came from:

1. Calling `get_object` for **every** `instance_detections2D` key (walls/floors).
2. Full-scene `ResetObjectFilter` every step.
3. Too many `PlaceObjectAtPoint` attempts.
4. Noisy `[displace]` prints.
5. Holding the full episode in RAM and building one giant pandas table only on `done`
   (episodes that hit the 1000-step horizon never called `save_data`, so only images existed).

Mitigations already in code:

- Cache the pickupable id set once. Nav and objects logging use one metadata reset for all FOV ids.
- Record each door's open/closed state once, at episode start.
- **`max_steps` hard cap** (same idea as an LLM context window): log at most the first
  `max_steps` frames; further frames are ignored (`Collector.at_capacity`).
  Non-positive `max_steps` (e.g. online eval’s temporary `-1`) means unset, not “already full”.
- **Flush on horizon**: `save_data(reason="done"|"max_steps")` so CSVs exist even when
  the agent never takes `done`.
- **Incremental CSV flush** every `flush_every` (default 50) steps for navigation /
  doors / object_state so RAM stays bounded.
- **FOV synthesis**: `renderInstanceSegmentation=True` at controller init, and
  per-step `renderImageSynthesis` is true on every agent step (stride 1). When
  synthesis is off, THOR may leave stale `instance_detections2D` on `last_event`.
  Bbox and mask columns are filled only when the task passes
  `include_detections=True` on a real synthesis step.

---

## 6. Bugs already fixed (do not reintroduce)

| Issue | Fix |
|-------|-----|
| `SPOCObject.get("pickupable", False)` always `False` | Implement `SPOCObject.get()` in `environment/spoc_objects.py` (builtin `dict.get` ignores `__getitem__`) |
| `PlaceObjectAtPoint(..., forceAction=True)` ValueError | Use `forceKinematic=True` |
| Objects left in the trial pose | `scripts/placement_trials.py` restores only after a successful place. A failed restore reloads the house before the next pickupable. |
| Scene name always `Procedural` | Use `house_<index>` |

---

## 7. How to run / review

```bash
# Note: tests/test_visibility_filters.py still targets the old silhouette /
# unoccluded-ratio API and needs a rewrite for visible-pixels / min-side /
# occupancy-ratio before it will pass against current collector.py.
```

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

```bash
export RUN="$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
export SCENE=house_XXXXXX   # from filenames

cat "$RUN/annotations/episode_meta-${SCENE}.json"
# Inspect placement_trials/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
```

Legacy spatial QA (`spatial_data_generation.py`, `qa_generator.py`) still consumes nav/objects CSVs (now under `annotations/`). Placement trials are the per-house `placement_candidates` file.

---

## 8. Building QA from placement trials

**Placement file:** `placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv` from `scripts/placement_trials.py`. `anchor-id` / `anchor-kind` name the receptacle or the landmark whose floor point was tried. `pos-*` is the resolved pose when `success` is true. A successful place is restored before the next anchor.

Do **not** expect egocentric direction columns at collection time; compute those later from an agent pose in the navigation CSV.

### Validator pattern

For a house, each pickupable has up to five receptacle rows and one floor row per landmark that has a spawn within the requested radius. `success` is true when `PlaceObjectAtPoint` succeeded. The original pose is restored only in that case. A failed restore reloads the house before the next pickupable.

### Example item types

1. **Final location:** “Where could the cup be?” → successful `placement_candidates` positions for that `obj-id`.
2. **Which surface:** group rows by `anchor-id`.
3. **Which object:** “Which object was tried on CounterTop|…?”

Example past run that used the old displacement register:  
`…/Generated/navigation/07_13_2026_15_15_34_072561/` (`house_007514`).

---

## 9. Survey side (lighter)

- `build_world_layout()` → rooms (polygon and area), doors as passages, landmark heuristics, connectivity.
- Agent room each step: `navigation-*.csv` columns `current-room`, `current-room-type`, `room-just-entered`.
- `passage_state` and `doors` record each door once, including `is_open`. RoomVisit does not open or close doors, so there is no change log.
- `nav_graph-*.json` from `GetReachablePositions` (start + end); see § output layout.

Survey “novel shortcut” validation is mostly **downstream** (cm-benchmark); collection provides layout + trajectory + reachability evidence.

---

## 10. Suggested next work (for agents)

- Optional: human-readable receptacle labels for templates.
- Writer that emits cm-benchmark items JSON from `placement_candidates` and `object_state` (including egocentric directions at a chosen query step).
- Do **not** invent semantic object facing from `obj-rot` unless THOR exposes a trusted signal.

---

## 11. Quick file map

| Path | Notes |
|------|--------|
| `collector.py` | Tracking, CSV/JSON export, `nav_graph` + `episode_meta` |
| `tasks/room_visit_task.py` | World layout, door log, nav graph snapshots |
| `scripts/placement_trials.py` | Per-house placement trials |
| `utils/nav_graph_export.py` | Reachable nodes + 8-connected edges |
| `scripts/summarize_episode_export.py` | Episode trajectory / nav summary |
| `environment/spoc_objects.py` | `.get()` fix |
| `configure_variables.sh` | Data dirs / navigation output |
| `README.md` | Short how-to-run; **this file** for full design context |
