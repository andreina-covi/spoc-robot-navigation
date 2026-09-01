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
    displacement_candidates-house_XXXXXX.csv
    displacement_debug-house_XXXXXX.csv
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

**Scene state:** start snapshot is taken at task init (before agent steps). End snapshot is re-queried at episode save (after door changes / displacements). Per-step door openness remains in `doors-*.csv` / `passage_state-*.csv`. Displacements that may alter later navigability are in `displacement_events-*.csv`. Prefer start for “map at rollout begin”; compare end if objects/doors moved.

**Assumptions / limitations:**

- Edges are **grid adjacency on the THOR reachability sample**, not proven single-action success for every SPOC heading (agent move is 0.2 m; sample spacing is 0.15 m). Use trajectory `action_success` for what actually happened.
- Two nearby Euclidean points are **not** connected unless they are adjacent on that grid.
- We do **not** export a heuristic of which nav edges the agent “knows”; only primitive FOV/visibility in `navigation` / `object_state`.
- Reachability can change mid-episode; start ≠ end when doors/objects move.

Summarize one run:

```bash
python scripts/summarize_episode_export.py --run_dir "$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
```

### Which objects go where

| Output | Objects included |
|--------|------------------|
| `navigation-*.csv` | **Named non-structural** FOV objects with `visible-pixels > 0`, **plus visibility metrics**. Drops Wall/Floor and numeric-only ids (e.g. `2|4`). Post-processing decides keep/drop. |
| `objects-*.csv` | Catalog of those FOV objects seen at least once (with instance color) |
| `object_state-*.csv` / displacement | **Pickupable** objects tracked after nav mask pixels at least once |
| `current-room` (in `navigation-*.csv`) | **Agent–room** membership (do **not** treat Floor/Wall as the room object) |

**Export vs filter (recommended policy):**

| Stage | What is dropped at collection time | Who decides the rest |
|-------|------------------------------------|----------------------|
| Navigation / objects CSV | Structural (Wall/Floor/…) + numeric-only ids (`2|4`) + zero mask pixels | Post-processing via metrics |
| Displacement `to_receptacle` pool | **Floor** and other structural receptacles (never sampled) | N/A |

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
| `displaced` | `True` if this object was successfully relocated earlier in the episode (still logged; filter in post-processing if needed) |

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
- `in_camera_fov` — **nav image** presence (instance mask pixels > 0 on synthesis
  strides; same idea as navigation `visible-pixels`). Used for displacement eligibility.
- `parent_receptacle`, `parent_receptacles`, `is_inside_receptacle`, `receptacle_is_open`

Tracking starts when a **pickupable** first has mask pixels in the agent camera.
HousePlant / Fridge / counters / windows stay in `navigation` only (not displace candidates).

**`displacement_events-*.csv`** (one row per accepted relocate; **two rows** share one `event_id` for a swap):

- `event_id`, `obj-id`, `at_timestep`, `from_receptacle`, `to_receptacle`, `from_pos-*`, `to_pos-*`
- `hidden_during` (must be `True` for accepted events), FOV flags, `moved_via`, `swap_partner_id`, `notes`
  - `moved_via=direct` — place onto a receptacle spawn
  - `moved_via=swap` — exchanged poses with `swap_partner_id` (different `objectType`)
  - `notes`: `same_receptacle_hidden_shift` | `other_receptacle_hidden_place` | `object_swap`
- A-not-B / original location for QA is **`from_pos-*`** (no extra column)

**`displacement_candidates-*.csv`** (three roles per accepted event when available):

| Column | Meaning |
|--------|---------|
| `event_id` | Joins to `displacement_events.event_id` |
| `obj-id`, `at_timestep` | Same object / step as the event |
| `candidate_role` | `chosen` \| `nearby_receptacle` \| `salient_decoy_location` |
| `candidate_receptacle` | Destination surface for that candidate |
| `candidate_pos-x/y/z` | Engine-resolved position after kinematic place |
| `is_persisted` | `True` only for `chosen` (real move left in the scene) |

Distractor rows are **trial teleports**: same `PlaceObjectAtPoint` + `forceKinematic` as the real move, position read back, then object restored to the chosen pose. Egocentric direction is **not** computed here (depends on later agent pose / query step).

**`navigation-*.csv`:** agent poses/rooms every step; **object rows only when
``visible-pixels > 0``** in that frame (named non-structural FOV objects, including
fridge / door / window). No padding of hidden tracked pickupables — those stay in
``object_state``. Optional ``displaced`` flag only appears on in-image rows.

**`objects-*.csv`:** one row per unique named object seen in the episode.
`bBox-center-*` / `size-*` are **3D AABB** fields from metadata (not segmentation).
They should be non-zero whenever THOR provides an AABB (independent of synthesis).

**`world_layout-*.json` / `passage_state`:** survey-oriented layout & room connectivity.

**`nav_graph-*.json`:** fine-grained THOR reachability (see Trajectory vs navigability above).

---

## 4. Invisible displacement algorithm (current)

Implemented in `RoomVisitTask.maybe_displace_hidden_objects` / `_try_hidden_place_on_receptacle`.

### Eligibility

1. Object is **pickupable** and had **nav mask pixels** (`visible-pixels > 0`) at least
   once — same signal as the RGB frames under `images/`, via `instance_detections2D`.
2. Absent from the nav image for **≥ 2** consecutive **synthesis** steps
   (`collector.candidates_for_displacement`; `in_camera_fov=False`).
3. Caps: `max_displacements=10` / episode, `1` displace **operation** / step
   (a swap logs two object rows but counts as one step operation; needs ≥2 remaining slots).
4. Displacement runs only on FOV synthesis strides (when mask/detections exist).
   Off-stride steps freeze `hidden_steps` (do not treat missing masks as “hidden”).
5. After repeated `place_failed`, that object’s `place_fail_count` rises so **other**
   eligible objects are preferred next step.

### Destination sampling

1. Candidate `to_receptacle` pool = non-pickupable receptacles in the agent's room,
   **excluding Floor / structural** meshes entirely (not sampled, not only discarded after fail).
2. Prefer spawn points ≥ `min_displace_distance` (0.25 m) from origin; try **same receptacle first**, then others in room.
3. Closed openables are skipped (`_receptacle_is_usable`).

### Object swap (fallback)

If receptacle place fails for the primary object:

1. Pick another eligible pickupable in the **same room**, **different `objectType`**, also
   **out of the nav image**, with ≥2 displacement slots remaining.
2. Kinematically **park A** (mid-air, or **Floor** spawn if another object sits
   between A and B), move B to A’s pose, then move A to B’s pose
   (a direct A→B place fails while B still occupies the destination).
   Floor is only a temporary hold — not a persisted `to_receptacle`.
3. Require **both** still out of image; validate positions; else restore both.
4. Persist **two** `displacement_events` rows sharing one `event_id`,
   `moved_via=swap`, `notes=object_swap`, mutual `swap_partner_id`.

### Realism rules (avoid “appears from nothing”)

1. Refuse if the object is in the **step nav event** mask set (`in_image_ids` —
   reuses that frame’s synthesis; no extra Pass).
2. `PlaceObjectAtPoint` with **`forceKinematic=True`** (this AI2-THOR Stretch build rejects `forceAction`).
3. Mid-place undoes use cheap THOR ``visible`` (no synthesis per spawn try).
4. **One** `Pass` + synthesis for the final mask check before persist (swap: one Pass
   for both objects). On mask hit → restore, no event row.
5. **Validate before persist:** read back object metadata; require position (and parent,
   when available) to match. On mismatch → restore, `stage=state_mismatch`.
6. Only **log** events that stay out of the nav image (`hidden_during=True`; `in_fov_just_after=False`).

### Distractor candidates (trial teleport)

After a validated real place (or swap), still in the scene at the chosen pose(s):

1. Pick **`nearby_receptacle`**: another usable open receptacle within ~1.5 m xz of the true destination.
2. Pick **`salient_decoy_location`**: largest-AABB-volume usable receptacle that is **not** near the true destination.
3. For each: kinematic place → read resolved pose → **restore** to the real destination. Only the chosen move stays in the scene.
4. Export all three (when available) to `displacement_candidates-*.csv` (per swapped object as well).

Requires `instance_detections2D` / `renderImageSynthesis` on the stride used for
displacement eligibility and post-place checks. Nav bbox metrics use the same stride.

### Tunables (`RoomVisitTask.__init__`)

- `max_displacements` (Collector), `max_displacements_per_step`
- `max_receptacles_to_try`, `max_place_coords`, `min_displace_distance`
- `NEARBY_RECEPTACLE_XZ_M` (module constant, default 1.5)
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
- Cache receptacles per room (**without Floor**).
- Limit place tries; quiet debug (CSV always; print mainly on success).
- Sample doors every `door_log_interval` steps.
- **`max_steps` hard cap** (same idea as an LLM context window): log at most the first
  `max_steps` frames; further frames are ignored (`Collector.at_capacity`).
  Non-positive `max_steps` (e.g. online eval’s temporary `-1`) means unset, not “already full”.
- **Flush on horizon**: `save_data(reason="done"|"max_steps")` so CSVs exist even when
  the agent never takes `done`.
- **Incremental CSV flush** every `flush_every` (default 50) steps for navigation /
  doors / object_state / displacement_debug so RAM stays bounded.
- **FOV synthesis stride**: `renderInstanceSegmentation=True` at controller init, but
  per-step `renderImageSynthesis` is `True` only every
  `stride = max(1, max_steps // CAP_PER_EPISODE)` agent steps for nav bbox / mask metrics
  **and** for displacement image-FOV tracking. With RoomVisit
  `CAP_PER_EPISODE = MAX_EPISODE_LEN`, stride is typically **1** (every step).
  Post-place: mid-loop uses THOR ``visible``; **one** mask Pass after a candidate
  place (or one Pass covering both objects after a swap).
  **Important:** when synthesis is off, THOR may leave *stale* `instance_detections2D`
  on `last_event`. Nav still writes `obj-id` / `obj-distance` every step from
  metadata; bbox / mask columns are filled only when the task passes
  `include_detections=True` on a real synthesis stride — never from leftover detections.
  Bbox metrics are read from the **navigation step event**, not `last_event` after
  displace `PlaceObjectAtPoint` (those steps run without synthesis and would drop
  detections).

---

## 6. Bugs already fixed (do not reintroduce)

| Issue | Fix |
|-------|-----|
| `SPOCObject.get("pickupable", False)` always `False` | Implement `SPOCObject.get()` in `environment/spoc_objects.py` (builtin `dict.get` ignores `__getitem__`) |
| `PlaceObjectAtPoint(..., forceAction=True)` ValueError | Use `forceKinematic=True` |
| Objects “appear” after displace | Undo if still in nav image (mask pixels); only accept out-of-image places |
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
# Inspect annotations/displacement_events / object_state tracks
```

Legacy spatial QA (`spatial_data_generation.py`, `qa_generator.py`) still consumes nav/objects CSVs (now under `annotations/`). **Invisible-displacement items** should be built from `displacement_events` + `object_state`, not only the older spatial JSON.

---

## 8. Building QA from displacements

**Ground-truth answer surface:** prefer `to_receptacle` / `from_receptacle` from **`displacement_events`**.  
`object_state.parent_receptacle` can disagree after kinematic place (THOR parenting quirks) — collection now rejects events when parent/position readback mismatches before persist.

Join **`displacement_candidates`** on `event_id` for multiple-choice positions:

- `chosen` — true destination (`is_persisted=True`)
- `nearby_receptacle` — nearby open surface distractor
- `salient_decoy_location` — large far surface distractor
- Original / A-not-B location — `from_pos-*` on the event row (not a candidate role)

For **`object_swap`** events, two rows share `event_id` and point at each other via
`swap_partner_id` (e.g. cup ↔ pepper). Each row’s `to_*` is the partner’s former pose.

Do **not** expect egocentric direction columns at collection time; compute those later from agent pose at the chosen query step.

### Validator pattern

For object `O` with event at `T`:

1. Some `t < T`: `in_camera_fov=True` (seen).
2. Some `t` with `T-ε ≤ t < T`: `in_camera_fov=False` still at `from_*` (hidden before move).
3. At `T`: pose/receptacle change; `hidden_during=True`; `in_fov_just_after=False`.
4. Last logged step for `O`: `in_camera_fov=False`.

### Example item types

1. **Final location:** “Where is the cup now?” → options from `displacement_candidates` positions / receptacles (+ `from_*` as A-not-B).  
2. **Same vs other surface:** use `notes` (`same_receptacle_hidden_shift` vs `other_receptacle_hidden_place`).  
3. **Which object:** “Which object was moved to CounterTop|…?”

`to_receptacle=Floor` is **not** produced by current collection (excluded from the candidate pool).

Example past run with 5 events:  
`…/Generated/navigation/07_13_2026_15_15_34_072561/` (`house_007514`).

---

## 9. Survey side (lighter)

- `build_world_layout()` → rooms, doors as passages, landmark heuristics, connectivity.  
- Agent room each step: `navigation-*.csv` columns `current-room`, `current-room-type`, `room-just-entered`.  
- `passage_state` derived from door logging (sparse in time).  
- `nav_graph-*.json` from `GetReachablePositions` (start + end); see § output layout.

Survey “novel shortcut” validation is mostly **downstream** (cm-benchmark); collection provides layout + trajectory + reachability evidence.

---

## 10. Suggested next work (for agents)

- Optional: human-readable receptacle labels for templates.  
- Writer that emits cm-benchmark items JSON from `displacement_events` + `displacement_candidates` + `object_state` (including egocentric directions at a chosen query step).  
- Door open/close events for survey door templates (`passage_events`).  
- Do **not** invent semantic object facing from `obj-rot` unless THOR exposes a trusted signal.

---

## 11. Quick file map

| Path | Notes |
|------|--------|
| `collector.py` | Tracking, CSV/JSON export, `nav_graph` + `episode_meta` |
| `tasks/room_visit_task.py` | Displacement + Floor-free receptacle pool + **object swap** + nav graph snapshots |
| `utils/nav_graph_export.py` | Reachable nodes + 8-connected edges |
| `scripts/summarize_episode_export.py` | Episode trajectory / nav summary |
| `environment/spoc_objects.py` | `.get()` fix |
| `configure_variables.sh` | Data dirs / navigation output |
| `README.md` | Short how-to-run; **this file** for full design context |
