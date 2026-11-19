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
| `invisible_displacement` | Per house, each movable object is trial-placed on floor points next to landmarks in its room and on the 5 largest non-openable receptacles in that room. Movable objects are every pickupable plus, with `placement.include_moveable`, every `moveable` object up to `moveable_max_mass_kg` with nothing resting on it (chairs, plants, lamps, bins). Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor, including Objaverse instances. Doors are excluded. Up to 3 floor spawns per landmark and up to 3 receptacle spawns nearest the top face. A receptacle place is kept only when that receptacle is in `parentReceptacles`. After a successful place the object is teleported back to its exact pose. A failed restore reloads the house before the next object. Rows land in the episode `annotations/` folder as `placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv` and include `obj-kind`, `radius_m` and `parent-receptacle`. |
| `survey_knowledge` | Multi-room trajectory + layout connectivity / door state |
| everything else | cm-benchmark builds all its constructs from the same episode folders (navigation, objects, poses, nav graph) |

Facing / intrinsic “front of object” is **out of scope** for this pass.

Episode kind tagged in meta: `invisible_displacement`. One RoomVisit command navigates, then trial-places objects in those houses (benchmark samples, or every house of a range with `--all_houses`; see §7). House indexes and each episode's `annotations/` path are passed in memory after eval. Navigation itself does not move objects.

---

## 2. Architecture

```text
online_eval (RoomVisit)
  └─ for each house in the eval samples (or each house of --house_start..--house_end with --all_houses)
       RoomVisitTask: agent navigates, Collector writes house_XXXXXX/images + annotations/
       worker returns annotations_dir with the sample result
  └─ after the agent finishes
       pass house indexes + those annotations/ paths to placement trials
       write placement_candidates-*.csv into each episode's annotations/
```

| Module | Role |
|--------|------|
| `tasks/room_visit_task.py` | World layout, door sampling, pickupable tracking, glance |
| `utils/glance.py` | Glance targets and the per-step choice of one target |
| `utils/placement_anchors.py` | Movable-object rule (`is_trial_object`), landmarks, receptacles, floor points, object rooms |
| `configs/collection.yaml` | Every collection setting: episode limits, glance, placement |
| `scripts/placement_trials.py` | Placement trials for the houses just navigated. Called at the end of the RoomVisit command. |
| `collector.py` | Visibility tracking, CSV/JSON writers, episode folder under `OBJAVERSE_NAVIGATION_PATH` |
| `environment/spoc_objects.py` | `SPOCObject.get()` fixed so `.get("pickupable", False)` works |

Scene naming: ProcTHOR `sceneName` is always `"Procedural"`. Files use `house_<zero-padded house_index>` from `task_info["house_index"]`.

---

## 3. Output layout

Path: `$OBJAVERSE_NAVIGATION_PATH/`  
(`configure_variables.sh` sets `OBJAVERSE_NAVIGATION_PATH`; `export_data.sh` uses `$EXPORT_OUT_ROOT/<split>/nav` instead).
Each episode folder is named after the house; a second episode of the same house gets `_2`, `_3`, ...
An unfinished folder that `export_data.sh` set aside before walking the house again is renamed `.partial-house_XXXXXX`.

```text
$OBJAVERSE_NAVIGATION_PATH/
  house_XXXXXX/                         # one episode
    images/
    annotations/
      navigation-house_XXXXXX.csv
      placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
      ...
```

Each episode root has **two** sibling folders:

```text
house_XXXXXX/
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
    placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
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
python scripts/summarize_episode_export.py --run_dir "$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX"
```

### Which objects go where

| Output | Objects included |
|--------|------------------|
| `navigation-*.csv` | Non-structural FOV objects with `visible-pixels > 0`, including numeric ids such as `6|5`. Each step has `ag-action`, `action_success` (`lastActionSuccess`), and the agent pose after that action. |
| `objects-*.csv` | Catalog of those FOV objects. `obj-id` is unchanged. `obj-type` and `obj-synset` are filled at export (`describe_object`). Also `assetId`, `room-id`, `room-type` (the room that contains the object, not the agent), the THOR properties `pickupable`, `moveable`, `mass`, and the axis-aligned box `aabb-min-*` / `aabb-max-*`. |
| `object_state-*.csv` | **Pickupable** objects tracked after nav mask pixels at least once |
| `current-room` (in `navigation-*.csv`) | **Agent–room** membership (do **not** treat Floor/Wall as the room object) |
| `obj-room`, `obj-room-type` (in `navigation-*.csv`) | Room id and room type of the **object**, from its position |

**Export vs filter (recommended policy):**

| Stage | What is dropped at collection time | Who decides the rest |
|-------|------------------------------------|----------------------|
| Navigation / objects CSV | Structural (Wall/Floor/Ceiling/Room) + zero mask pixels. Numeric ids are kept. | cm-benchmark, from the metrics |

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
| `mark-x`, `mark-y` | Full-image pixel farthest from the edge of that same mask crop |
| `in-frame-fraction` | Share of the expected (unoccluded) box inside the frame; 0 when the box is cut by the camera plane or cannot be projected |

No cutoff is applied here; cm-benchmark holds the keep/drop rules.

Fully hidden / behind other geometry never appear in detections. Out-of-camera objects are
not in `navigation` for that step. Hidden pickupables appear only in `object_state`
(after they were tracked via mask pixels at least once).

**Spatial relations (post-processing):**
- Agent ↔ object: use `navigation` rows + agent pose vs object pose / bbox.
- Agent ↔ room: use `current-room` / `current-room-type` (and `world_layout`), not structural mesh rows.
- Object ↔ room: use `obj-room` / `obj-room-type` on the navigation row, or `room-id` / `room-type` in `objects-*.csv`. These follow the object's position, which can differ from the agent's room.
- Agent ↔ navigable map: snap poses to `nav_graph` nodes; do **not** replace the trajectory with a shortest path on that graph.

### Important columns

**`object_state-*.csv`** (per timestep × tracked pickupable; includes hidden rows):

- `timestep`, `obj-id`, `obj-type`, pose
- `visible` — THOR metadata (`visibilityDistance` LOS)
- `in_camera_fov` — nav-image presence when a mask was rendered.
- `parent_receptacle`, `parent_receptacles`, `is_inside_receptacle`, `receptacle_is_open`

Tracking starts when a **pickupable** first has mask pixels in the agent camera.
HousePlant / Fridge / counters / windows stay in `navigation` only.

**`placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`** is written into that episode's `annotations/` folder by `scripts/placement_trials.py` after navigation. `obj-kind` is `pickupable` or `moveable` (a non-pickupable object such as a chair). `anchor-id` is the receptacle id or the landmark id. `anchor-kind` is `receptacle` or `floor`. `pos-*` is the resolved pose when `success` is true. A successful place is restored before the next anchor. Episodes collected before this change may still contain `displacement_events`, `displacement_candidates`, and `displacement_debug`; those files are not written now.

**`navigation-*.csv`:** one row group per step with `ag-action`, `action_success` (`lastActionSuccess`), and `ag-pos-*` / `ag-rot-*` after the action. `current-room` and `current-room-type` are the agent's room. `obj-room` and `obj-room-type` are the room that contains that object. Object rows are added only when `visible-pixels > 0`. Numeric ids are included. Hidden tracked pickupables stay in `object_state`.
`scripted-glance` is true on frames inserted by the open-loop glance (not chosen by the policy), and `glance-obj-id` names the object looked at. `step-source` says who chose each step: `policy`, `glance`, or `done_override` (a turn that replaced an early `done`). Consumers that replay the agent's route (cm-benchmark class 4) must skip glance frames.

**`episode_meta-*.json`:** `collection_config` is the `configs/collection.yaml` used for the run. `glances` lists each glance (`object_id`, `policy_step`, `first_timestep`, `turns`, `bearing_deg`, `distance_m`, `expected_px`). `save_reason` is `done`, `max_steps` or `timeout`, and `truncated` is true on a timeout. `done_overrides` counts the `done` actions turned into a turn because rooms were still unvisited.

**`objects-*.csv`:** one row per unique exported object. `obj-id` is the THOR id, including values like `6|5`. `obj-type` and `obj-synset` come from `describe_object`. `aabb-min-*` / `aabb-max-*` are the 3D axis-aligned corners. `room-id` and `room-type` are the room that contains the object. `pickupable`, `moveable` and `mass` are the THOR metadata values. Objaverse assets report mass 1.0, so use the box size to judge how large they are.

**`world_layout-*.json`:** each region has `polygon` (floor outline) and `area` (square meters on xz). `doors` and `passage_state` record each door once at the start, including whether it is open.

**`nav_graph-*.json`:** fine-grained THOR reachability (see Trajectory vs navigability above).

---

## 4. Placement trials

The RoomVisit command already knows the house indexes (`house_id` on the eval samples) and each episode's `annotations/` folder (returned by the worker). After the agent finishes, that same command passes both to the placement trials. Nothing is written just to remember the indexes.

The CSV is written next to the other episode files: `$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX/annotations/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv`. The radius is `--floor_radius_m` on the RoomVisit command (required; there is no default). Running `scripts/placement_trials.py` alone still accepts `--output` (default `$OBJAVERSE_NAVIGATION_PATH/placement_trials`) for a manual directory.

The objects tried are every pickupable and, when `placement.include_moveable` is true, every `moveable` object with mass up to `moveable_max_mass_kg` and nothing resting on it. Pickupables alone are too small for the ID construct: in Round 4 (10 houses) only 22 of 155 placed-and-seen pickupables ever covered 1% of the frame. Doors and structure are never moved. An object that carries other objects is skipped, because moving it would let them fall.

For each of those objects:

1. Floor: up to `max_floor_spawns` floor spawns next to each landmark in its room, nearest to the landmark's box first and at least `min_spawn_separation_m` apart. A spawn qualifies when its gap to the landmark's box footprint is at least the object's half footprint plus `floor_clearance_margin_m` and at most the radius, so the object lands beside the landmark and not inside it. The first point where the place succeeds is kept. Landmarks are the 5 largest non-pickupable, non-structural objects whose box bottom is within 0.1 m of the floor height, including Objaverse instances. Doors and doorways are excluded, and so is the object itself. A room with no floor uses the lowest floor height in the house. `--top_landmarks` changes that count. Object poses are read after 10 physics steps.
2. Receptacles: the 5 largest non-openable receptacles in that room (`receptacle_salience` is the axis-aligned box volume). Floor, wall, ceiling, room, and openable receptacles are excluded. A non-pickupable object that stands on the floor (a chair, a floor lamp) gets floor trials only.
3. Up to `max_receptacle_spawns` spawn coordinates per receptacle, closest to the center of its top face and at least `min_spawn_separation_m` apart (adjacent points are all blocked by the same object). Spawn lists are cached per receptacle.
4. `PlaceObjectAtPoint` with `forceKinematic`. It places the object's position (pivot), not its bottom, so every target is raised by the object's pivot-to-bottom height plus `lift_epsilon_m`; without this the object intersects the surface ("Spawn area not clear"). Settings are under `placement:` in `configs/collection.yaml`. A receptacle row is `success` only when that receptacle is in `parentReceptacles` after the place. `parent-receptacle` records those ids.
5. Restore: `TeleportObject` with `forceAction` puts the object back at its exact original pose. `PlaceObjectAtPoint` cannot do this when a neighbour touches the original pose. If the object is farther than `restore_tolerance_m` from its origin afterwards, the house is reloaded before the next object. If a house raises an error, the controller is restarted and the run goes on with the next house; that house gets no placement file.
6. Columns: `obj-id`, `obj-kind` (`pickupable` or `moveable`), `anchor-id`, `anchor-kind` (`receptacle` or `floor`), `pos-*`, `success`, `radius_m`, `parent-receptacle`. The split and radius are also in the file name.

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
- **Open-loop glance** (`utils/glance.py`, settings in `configs/collection.yaml`):
  the targets are the placement-trial objects (`glance_candidates`): every pickupable
  and, with `placement.include_moveable`, every `moveable` object up to
  `moveable_max_mass_kg` with nothing resting on it. An object inside a closed
  receptacle is not a target. In the ten Round 4 walks, where the glance looked at
  pickupables only, 36 of 75 placed `moveable` objects never came into view.
  Positions and sizes are read once at episode start. Each step runs one
  numpy distance/bearing computation, with no rendering and no simulator call.
  A target must be in the agent's room, out of view, within `max_distance_m`, and
  large enough to cover `area_margin` × `min_frame_fraction` of the frame from there.
  When several qualify, the one with the largest expected area is chosen.
  A glance turns a fixed number of 45° steps toward one object (at most `max_turns`),
  then turns back. It never loops until the object is visible; the 1% frame check
  happens offline in cm-benchmark. Limits: one glance per object, `max_per_episode`,
  `min_steps_between`. Glance frames are added to the `max_steps` frame budget.
- **Wall-clock timeout** (`episode.timeout_s`): the episode saves with
  `save_reason="timeout"` and ends.
- **Policy-step cap** (`episode.max_policy_steps`, 600): clamps the task horizon
  (RoomVisit 1000) so long walks do not slow cm-benchmark generation; the episode
  saves with `save_reason="max_steps"`. Glance frames are extra. `null` keeps the horizon.
- **Room coverage** (`episode.continue_until_all_rooms_visited`): while a room of the
  house has not been visited (the agent's position, as in `visited-rooms`), the
  policy's `done` is replaced by one `done_override_action` (a 45° turn), so the
  policy re-plans from a new view. After `max_done_overrides` (16) ignored `done`s
  the next one is accepted; the step cap and timeout still apply. `episode_meta`
  records the count as `done_overrides`.
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
| Objects left in the trial pose | `scripts/placement_trials.py` restores only after a successful place. A failed restore reloads the house before the next object. |
| Scene name always `Procedural` | Use `house_<index>` |
| Room floors got the wrong room or none. A ProcTHOR room floor has the room id as its `objectId` and position (0, 0, 0). Houses whose origin is outside every room exported no landmarks and no floor placements (3 of 10 in Round 4); the others had floor placements in one room only. | `object_room_id` in `utils/placement_anchors.py` reads a floor's room from its id. Only the house-wide `Floor` returns spawn points, so `floor_spawns_by_room` assigns each point to the room whose polygon contains it |
| Restoring a trial object with `PlaceObjectAtPoint` failed when a neighbour touched its original pose | `restore_object` uses `TeleportObject` with `forceAction` |

---

## 7. How to run / review

Tests: `python -m pytest tests` (after `. configure_variables.sh`). They need no simulator.

`configure_variables.sh` holds this machine's paths and is not in git (`.gitignore`). Create it once per machine from the template:

```bash
cp configure_variables.example.sh configure_variables.sh   # then edit the paths
```

Benchmark samples (a few episodes; `$LOGS_DIR` and `$CKPT_DIR` are your own folders for eval logs and the checkpoint):

```bash
. configure_variables.sh

python -m training.offline.online_eval --shuffle --eval_subset minival \
  --output_basedir "$LOGS_DIR" \
  --test_augmentation --task_type RoomVisit \
  --eval_set_size 1 \
  --floor_radius_m 1.2 \
  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
  --house_set objaverse --wandb_logging False --num_workers 1 \
  --gpu_devices 0 \
  --training_run_id SigLIP-ViTb-3-CHORES-S \
  --local_checkpoint_dir "$CKPT_DIR"
```

Every house of a split, in resumable chunks (`END` not set = every house of the split; full settings and resume rules in the README, "Export every house of a split"):

```bash
SPLIT=val START=0 END=3 ./export_data.sh    # test on 3 houses; never `source` it
SPLIT=val CHUNK=500 ./export_data.sh        # all 16,000 val houses; rerun to resume
```

It calls `online_eval` with `--all_houses --house_start --house_end`: one RoomVisit sample per house (goal text as in the benchmark; the start is the first reachable position and heading, in an order seeded by the house index, where the agent can be teleported), then placement trials for the chunk. Finished houses (`episode_meta-*.json`) are skipped on a rerun (`utils/export_progress.py`). The output folder keeps only the episode folders and one log per chunk.

```bash
export RUN="$OBJAVERSE_NAVIGATION_PATH/house_XXXXXX"
export SCENE=house_XXXXXX   # from filenames

cat "$RUN/annotations/episode_meta-${SCENE}.json"
# Inspect $RUN/annotations/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv
```

---

## 8. Using the placement trials (cm-benchmark)

**Placement file:** `$RUN/annotations/placement_candidates-house_XXXXXX-<split>-radius_<meters>.csv` from `scripts/placement_trials.py`. `anchor-id` / `anchor-kind` name the receptacle or the landmark whose floor point was tried. `pos-*` is the resolved pose when `success` is true. `parent-receptacle` lists parent ids after a receptacle place. A successful place is restored before the next anchor.

Do **not** expect egocentric direction columns at collection time; compute those later from an agent pose in the navigation CSV.

### Validator pattern

For a house, each movable object (`obj-kind` `pickupable` or `moveable`) has one floor row per landmark that has a qualifying spawn within the requested radius. It also has up to five receptacle rows, except a non-pickupable object that stands on the floor, which has floor rows only. Floor `success` is true when `PlaceObjectAtPoint` succeeded at one of up to `max_floor_spawns` points. A floor row means the object lies on the floor beside `anchor-id`, not on it. Receptacle `success` also requires that receptacle id in `parentReceptacles` (up to 3 nearest top-face spawns are tried). The original pose is restored only when the place action succeeds. A failed restore reloads the house before the next object.

cm-benchmark reads this file in stage 1 and uses the successful places as destinations for invisible-displacement items.
---

## 9. Survey side (lighter)

- `build_world_layout()` → rooms (polygon and area), doors as passages, landmark heuristics, connectivity.
- Agent room each step: `navigation-*.csv` columns `current-room`, `current-room-type`, `room-just-entered`.
- Object room: `obj-room` and `obj-room-type` on each navigation object row, and `room-id` / `room-type` in `objects-*.csv`. These come from the object's position.
- `passage_state` and `doors` record each door once, including `is_open`. RoomVisit does not open or close doors, so there is no change log.
- `nav_graph-*.json` from `GetReachablePositions` (start + end); see § output layout.

Survey “novel shortcut” validation is mostly **downstream** (cm-benchmark); collection provides layout + trajectory + reachability evidence.

---

## 10. Suggested next work (for agents)

- Items are built in cm-benchmark, not here. If a quality fix there needs a field this export lacks, add it to the collector and collect again.
- Do **not** invent semantic object facing from `obj-rot` unless THOR exposes a trusted signal.

---

## 11. Quick file map

| Path | Notes |
|------|--------|
| `collector.py` | Tracking, CSV/JSON export, `nav_graph` + `episode_meta` |
| `training/offline/online_eval.py` | RoomVisit runs; `--all_houses` walks a range of houses |
| `tasks/multi_task_eval_sampler.py` | Seeded start pose for `--all_houses` (`seeded_start_candidates`) |
| `configs/collection.yaml` | Every collection setting |
| `tasks/room_visit_task.py` | World layout, door log, nav graph snapshots |
| `scripts/placement_trials.py` | Per-house placement trials |
| `utils/nav_graph_export.py` | Reachable nodes + 8-connected edges |
| `scripts/summarize_episode_export.py` | Episode trajectory / nav summary |
| `environment/spoc_objects.py` | `.get()` fix |
| `configure_variables.example.sh` | Template for the local, git-ignored `configure_variables.sh` (data dirs, navigation output, export settings) |
| `export_data.sh` | Chunked, resumable export of every house of a split |
| `utils/export_progress.py` | Finished / unfinished episodes of a collection folder (resume) |
| `README.md` | Short how-to-run; **this file** for full design context |
