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
| `invisible_displacement` | Object seen → leaves **nav** FOV → relocated **while still hidden** → still not in final nav frame; true final receptacle/pose in logs |
| `survey_knowledge` | Multi-room trajectory + layout connectivity / door state |

Facing / intrinsic “front of object” is **out of scope** for this pass.

Episode kind tagged in meta: `invisible_displacement` (RoomVisit still does normal nav; displacement is opportunistic during the same run).

---

## 2. Architecture

```text
online_eval (RoomVisit)
  └─ RoomVisitTask._step
       ├─ agent_step (nav)
       ├─ log visible non-structural FOV objects → navigation / objects CSV
       ├─ track pickupables only → object_state + maybe_displace_hidden_objects()
       ├─ Collector.collect_data(...)   # capped at max_steps; flushes every 50 steps
       └─ on done OR max_steps → Collector.save_data()
```

| Module | Role |
|--------|------|
| `tasks/room_visit_task.py` | Displacement policy, world layout, door sampling interval, pickupable caches |
| `collector.py` | Visibility tracking, CSV/JSON writers, episode folder under `OBJAVERSE_NAVIGATION_PATH` |
| `environment/spoc_objects.py` | `SPOCObject.get()` fixed so `.get("pickupable", False)` works |

Scene naming: ProcTHOR `sceneName` is always `"Procedural"`. Files use `house_<zero-padded house_index>` from `task_info["house_index"]`.

---

## 3. Output layout

Path: `$OBJAVERSE_NAVIGATION_PATH/<timestamp>/`  
(`configure_variables.sh` sets `OBJAVERSE_NAVIGATION_PATH`).

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
    displacement_events-house_XXXXXX.csv
    displacement_debug-house_XXXXXX.csv
    passage_state-house_XXXXXX.csv
    region_trajectory-house_XXXXXX.csv
    world_layout-house_XXXXXX.json
    episode_meta-house_XXXXXX.json
```

**Do not** repeat `episode_id` / `scene_id` / `episode_kind` on every CSV row. Those live in `annotations/episode_meta-*.json` and the folder name. Consumers should join on folder + `timestep` / `obj-id`. Navigation `path` columns still point at files under `images/`.

`episode_meta-*.json` also stores run geometry and agent constants needed for offline recomputation:

| Section | Fields |
|---------|--------|
| `camera` | `width`, `height`, `frame_size_px`, `fov_vertical_deg` (nav / INTEL) |
| `agent` | `movement_constant`, `rotation_deg`, `horizon_deg`, `arm_move_constant`, `wrist_rotation_deg` |
| `visibility_filters` | mode, export policy, suggested post-process thresholds |

### Which objects go where

| Output | Objects included |
|--------|------------------|
| `navigation-*.csv` | **Named non-structural** objects with any nav `instance_detections2D` pixels this step, **plus visibility metrics**. Drops Wall/Floor and numeric-only ids (e.g. `2|4`). Post-processing decides keep/drop. |
| `objects-*.csv` | Catalog of those FOV objects seen at least once (with instance color) |
| `object_state-*.csv` / displacement | **Pickupable only**, and only after a **recognizable** first sighting (`passes_visibility_filters`) |
| `current-room` / `region_trajectory` | **Agent–room** membership (do **not** treat Floor/Wall as the room object) |

**Export vs filter (recommended policy):**

| Stage | What is dropped at collection time | Who decides the rest |
|-------|------------------------------------|----------------------|
| Navigation / objects CSV | Structural (Wall/Floor/…) + numeric-only ids (`2|4`) | Post-processing via metrics |
| Displacement “seen” | Tiny / thin / low shown-% peeks | Collector (`filter_recognizable_detections`) |

**Why export-all-with-metrics for nav:** thresholds can be retuned without re-running THOR; different tasks can use different cutoffs.

**Visibility metrics on each navigation object row** (formulas and caveats:
**[VISIBILITY_METRICS.md](VISIBILITY_METRICS.md)**):

| Column | Meaning |
|--------|---------|
| `visible-area-px` | Visible bbox area (or mask pixels in strict mode) |
| `visible-frac` | `visible-area-px / frame_size` |
| `full-silhouette-px` | Estimated full object screen area (size÷distance) |
| `unoccluded-ratio` | `visible / full-silhouette` (shown % of the object) |

Suggested post-process cutoffs are stored per run in
`episode_meta.visibility_filters.suggested_postprocess_thresholds`
(see [VISIBILITY_METRICS.md](VISIBILITY_METRICS.md) for current code defaults).

Example post-process keep rule: apply your own cutoffs on
`visible-area-px` / `unoccluded-ratio` (or use the suggested thresholds from episode_meta).

Fully hidden / behind other geometry never appear in detections. Out-of-camera objects are
not in `navigation` for that step. Hidden pickupables still appear in `object_state` after
they were tracked.

**Spatial relations (post-processing):**
- Agent ↔ object: use `navigation` rows + agent pose vs object pose / bbox.
- Agent ↔ room: use `current-room` / `region_trajectory` (and `world_layout`), not structural mesh rows.

### Important columns

**`object_state-*.csv`** (per timestep × tracked pickupable; includes hidden rows):

- `timestep`, `obj-id`, `obj-type`, pose, `visible` (THOR), `in_camera_fov` (nav `instance_detections2D`)
- `parent_receptacle`, `parent_receptacles`, `is_inside_receptacle`, `receptacle_is_open`

**`displacement_events-*.csv`** (one row per accepted relocate):

- `obj-id`, `at_timestep`, `from_receptacle`, `to_receptacle`, `from_pos-*`, `to_pos-*`
- `hidden_during` (must be `True` for accepted events), FOV flags, `notes`
  - `same_receptacle_hidden_shift` — same surface, different pose (e.g. counter left→right)
  - `other_receptacle_hidden_place` — different receptacle in the same room

**`navigation-*.csv`:** agent poses, rooms, **named non-structural** FOV objects + bbox,
`visible-area-px`, `visible-frac`, `full-silhouette-px`, `unoccluded-ratio`,
`action_success`, `held_obj-id`.

**`world_layout-*.json` / `passage_state` / `region_trajectory`:** survey-oriented layout & room path.

---

## 4. Invisible displacement algorithm (current)

Implemented in `RoomVisitTask.maybe_displace_hidden_objects` / `_try_hidden_place_on_receptacle`.

### Eligibility

1. Object is **pickupable** and was seen at least once in **nav** `instance_detections2D`.
2. Out of nav FOV for **≥ 2** consecutive steps (`collector.candidates_for_displacement`).
3. Caps: `max_displacements=5` / episode, `1` displace / step.

### Realism rules (avoid “appears from nothing”)

1. Refuse if still in nav detections before place.
2. `PlaceObjectAtPoint` with **`forceKinematic=True`** (this AI2-THOR Stretch build rejects `forceAction`).
3. Prefer spawn points ≥ `min_displace_distance` (0.25 m) from origin; try **same receptacle first**, then others in room.
4. After each place: refresh instance segmentation; if object is **in nav FOV**, **undo** (place back) and try another point.
5. Only **log** events that stay off-camera (`hidden_during=True`). Never keep a move that pops into the current nav image.

### Tunables (`RoomVisitTask.__init__`)

- `max_displacements` (Collector), `max_displacements_per_step`
- `max_receptacles_to_try`, `max_place_coords`, `min_displace_distance`
- `door_log_interval` (default 5)

---

## 5. Performance constraints (do not regress)

Earlier freezes came from:

1. Calling `get_object` for **every** `instance_detections2D` key (walls/floors).
2. Full-scene `ResetObjectFilter` every step.
3. Too many `PlaceObjectAtPoint` attempts.
4. Door metadata every step + noisy `[displace]` prints.
5. Holding the full episode in RAM and building one giant pandas table only on `done`
   (episodes that hit the 1000-step horizon never called `save_data`, so only images existed).

Mitigations already in code:

- Cache pickupable id set once for displacement; nav/objects logging uses one metadata reset for all FOV ids.
- Cache receptacles per room.
- Limit place tries; quiet debug (CSV always; print mainly on success).
- Sample doors every `door_log_interval` steps.
- **`max_steps` hard cap** (same idea as an LLM context window): log at most the first
  `max_steps` frames; further frames are ignored (`Collector.at_capacity`).
  Non-positive `max_steps` (e.g. online eval’s temporary `-1`) means unset, not “already full”.
- **Flush on horizon**: `save_data(reason="done"|"max_steps")` so CSVs exist even when
  the agent never takes `done`.
- **Incremental CSV flush** every `flush_every` (default 50) steps for navigation /
  doors / object_state / region_trajectory / displacement_debug so RAM stays bounded.

---

## 6. Bugs already fixed (do not reintroduce)

| Issue | Fix |
|-------|-----|
| `SPOCObject.get("pickupable", False)` always `False` | Implement `SPOCObject.get()` in `environment/spoc_objects.py` (builtin `dict.get` ignores `__getitem__`) |
| `PlaceObjectAtPoint(..., forceAction=True)` ValueError | Use `forceKinematic=True` |
| Objects “appear” after displace | Undo if still in nav FOV; only accept hidden places |
| Scene name always `Procedural` | Use `house_<index>` |

---

## 7. How to run / review

```bash
# Unit tests for visibility filters
python -m pytest tests/test_visibility_filters.py -v
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
# Inspect annotations/displacement_events / object_state tracks
```

Legacy spatial QA (`spatial_data_generation.py`, `qa_generator.py`) still consumes nav/objects CSVs (now under `annotations/`). **Invisible-displacement items** should be built from `displacement_events` + `object_state`, not only the older spatial JSON.

---

## 8. Building QA from displacements

**Ground-truth answer surface:** prefer `to_receptacle` / `from_receptacle` from **`displacement_events`**.  
`object_state.parent_receptacle` can disagree after kinematic place (THOR parenting quirks).

### Validator pattern

For object `O` with event at `T`:

1. Some `t < T`: `in_camera_fov=True` (seen).
2. Some `t` with `T-ε ≤ t < T`: `in_camera_fov=False` still at `from_*` (hidden before move).
3. At `T`: pose/receptacle change; `hidden_during=True`; `in_fov_just_after=False`.
4. Last logged step for `O`: `in_camera_fov=False`.

### Example item types

1. **Final location:** “Where is the cup now?” → options include `to_receptacle` vs `from_receptacle` vs other surfaces.  
2. **Same vs other surface:** use `notes` (`same_receptacle_hidden_shift` vs `other_receptacle_hidden_place`).  
3. **Which object:** “Which object was moved to CounterTop|…?”

`to_receptacle=Floor` is valid but weaker for natural questions; optionally filter when sampling items.

Example past run with 5 events:  
`…/Generated/navigation/07_13_2026_15_15_34_072561/` (`house_007514`).

---

## 9. Survey side (lighter)

- `build_world_layout()` → rooms, doors as passages, landmark heuristics, connectivity.  
- `region_trajectory` from agent room each step.  
- `passage_state` derived from door logging (sparse in time).  

Survey “novel shortcut” validation is mostly **downstream** (cm-benchmark); collection provides layout + trajectory evidence.

---

## 10. Suggested next work (for agents)

- Prefer non-`Floor` destinations (or bias spawn toward tables/counters).  
- Optional: human-readable receptacle labels for templates.  
- Writer that emits cm-benchmark items JSON from `displacement_events` + `object_state`.  
- Door open/close events for survey door templates (`passage_events`).  
- Do **not** invent semantic object facing from `obj-rot` unless THOR exposes a trusted signal.

---

## 11. Quick file map

| Path | Notes |
|------|--------|
| `collector.py` | Tracking, CSV/JSON export, `episode_meta` |
| `tasks/room_visit_task.py` | Displacement + layout + RoomVisit `_step` hooks |
| `environment/spoc_objects.py` | `.get()` fix |
| `configure_variables.sh` | Data dirs / navigation output |
| `README.md` | Short how-to-run; **this file** for full design context |
