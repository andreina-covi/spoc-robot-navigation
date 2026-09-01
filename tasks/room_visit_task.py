from typing import Any, Dict, List, Optional, Set
import random

import numpy as np
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from shapely.geometry import Point
from typing_extensions import Literal

from environment.stretch_controller import StretchController
from tasks.abstract_task import AbstractSPOCTask
from utils.distance_calculation_utils import position_dist
from utils.type_utils import RewardConfig, THORActions
from utils.constants.object_constants import is_exportable_object, is_structural_object
from utils.constants.stretch_initialization_utils import (
    AGENT_MOVEMENT_CONSTANT,
    AGENT_ROTATION_DEG,
)
from utils.nav_graph_export import (
    build_nav_graph_from_reachable_positions,
    reachability_grid_size,
)
from training.online.reward.reward_shaper import RoomVisitRewardShaper
from collector import Collector
from online_evaluation.max_episode_configs import MAX_EPISODE_LEN_PER_TASK

CAP_PER_EPISODE = MAX_EPISODE_LEN_PER_TASK["RoomVisit"]

# xz distance (m): receptacles within this of the chosen destination count as "nearby"
NEARBY_RECEPTACLE_XZ_M = 1.5

class RoomVisitTask(AbstractSPOCTask):
    task_type_str = "RoomVisit"

    def __init__(
        self,
        controller: StretchController,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_names: List[str],
        reward_config: Optional[RewardConfig] = None,
        distance_type: Literal["l2"] = "l2",
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self.last_taken_action_str = ""
        self.last_action_success = -1
        self.last_action_random = -1

        self.reachable_positions = controller.get_reachable_positions()
        self.seen_rooms = []

        self.last_num_seen_rooms = len(self.seen_rooms)

        self.distance_type = distance_type
        self.dist_to_target_func = self.min_l2_distance_to_target

        last_distance = self.dist_to_target_func()
        self.closest_distance = last_distance
        self.optimal_distance = (
            last_distance
            if self.dist_to_target_func == self.min_geodesic_distance_to_target
            else self.min_geodesic_distance_to_target()
        )

        self.visualize = visualize
        if reward_config is not None:
            self.reward_shaper = RoomVisitRewardShaper(task=self)
        else:
            self.reward_shaper = None

        self.num_sub_done = 0
        self.num_successful_sub_done = 0
        self._took_sub_done_action = False
        self.visited_rooms = set()
        self.visited_loc = set()
        # ProcTHOR metadata sceneName is always "Procedural"; identify by house_index
        house_index = self.task_info.get("house_index", "unknown")
        scene_name = f"house_{str(house_index).zfill(6)}"
        # Online eval may pass max_steps=-1 at construction, then patch task.max_steps
        # later. Treat non-positive as unset; _step syncs the real horizon.
        collector_max_steps = max_steps if (max_steps is not None and max_steps > 0) else None
        self.collector = Collector(
            scene_name=scene_name,
            episode_kind="invisible_displacement",
            max_displacements=10,
            max_steps=collector_max_steps,
            flush_every=50,
        )
        self.collector.set_world_layout(self.build_world_layout())
        self._export_nav_graph_snapshot("episode_start")
        self._displacements_this_step = 0
        self.max_displacements_per_step = 1
        self.max_receptacles_to_try = 4
        self.max_place_coords = 4
        self.min_displace_distance = 0.25  # meters; same table left↔right is OK
        self.door_log_interval = 5  # log doors every N steps (not every step)
        self._pickupable_ids: Optional[Set[str]] = None
        self._receptacles_by_room: Dict[str, List[Dict[str, Any]]] = {}
        self._last_door_states = None
        # FOV instance synthesis every ``stride`` agent steps (see ``_step``).
        if collector_max_steps is not None and collector_max_steps > 0:
            self.stride = max(1, int(collector_max_steps) // CAP_PER_EPISODE)
        else:
            self.stride = 1
        self._sync_collector_horizon()

    def _sync_collector_horizon(self):
        """Keep collector max_steps / FOV stride / nav camera params in sync."""
        if self.max_steps is not None and self.max_steps > 0:
            self.collector.max_steps = self.max_steps
            self.stride = max(1, int(self.max_steps) // CAP_PER_EPISODE)
        self.collector.fov_stride = self.stride
        # Actual RotateCameraMount / FOV from StretchController.calibrate_agent
        mount = getattr(self.controller, "nav_camera_mount_deg", None)
        fov = getattr(self.controller, "nav_camera_fov_deg", None)
        if mount is not None:
            self.collector.nav_camera_mount_deg = float(mount)
        if fov is not None:
            self.collector.nav_camera_fov_deg = float(fov)

    def build_world_layout(self) -> Dict[str, Any]:
        """Survey-knowledge layout: regions, landmarks, passages, connectivity."""
        regions = []
        for room in self.house.get("rooms", []):
            poly = room.get("floorPolygon", [])
            if poly:
                cx = float(np.mean([p["x"] for p in poly]))
                cz = float(np.mean([p["z"] for p in poly]))
                cy = float(np.mean([p.get("y", 0.0) for p in poly]))
            else:
                cx = cy = cz = 0.0
            regions.append(
                {
                    "region_id": room["id"],
                    "label": room.get("roomType", room["id"]),
                    "center": {"x": cx, "y": cy, "z": cz},
                    "landmark_obj_ids": [],
                }
            )

        passages = []
        connectivity = []
        for door in self.house.get("doors", []):
            passage = {
                "passage_id": door["id"],
                "obj-id": door["id"],
                "from_region": door.get("room0"),
                "to_region": door.get("room1"),
                "passage_type": "door",
            }
            passages.append(passage)
            connectivity.append(
                {
                    "from_region": door.get("room0"),
                    "to_region": door.get("room1"),
                    "passage_id": door["id"],
                    "bidirectional": True,
                }
            )

        # Landmarks: prefer large static house objects if present
        landmarks = []
        landmark_types = {
            "Fridge",
            "Television",
            "Sofa",
            "Bed",
            "Toilet",
            "Sink",
            "DiningTable",
            "CoffeeTable",
            "SideTable",
            "CounterTop",
            "Microwave",
            "Oven",
            "Desk",
            "Dresser",
            "ArmChair",
            "TelevisionStand",
        }
        for obj in self.house.get("objects", []):
            otype = obj.get("assetId", "") or obj.get("id", "")
            # house JSON uses nested structure; also try objectType-like fields
            obj_type = obj.get("objectType") or obj.get("assetId", "").split("_")[0]
            if obj_type not in landmark_types and not any(
                t.lower() in str(otype).lower() for t in landmark_types
            ):
                continue
            pos = obj.get("position", {"x": 0, "y": 0, "z": 0})
            room_id, _ = None, None
            try:
                room_id, _ = self.controller.get_objects_room_id_and_type(obj["id"])
            except Exception:
                room_id = None
            landmarks.append(
                {
                    "landmark_id": obj.get("id"),
                    "obj-type": obj_type,
                    "position": pos,
                    "region_id": room_id,
                }
            )
            if room_id is not None:
                for r in regions:
                    if r["region_id"] == room_id:
                        r["landmark_obj_ids"].append(obj.get("id"))
                        break

        return {
            "regions": regions,
            "landmarks": landmarks,
            "passages": passages,
            "connectivity": connectivity,
            "nav_metadata": {
                "grid_size": reachability_grid_size(),
                "agent_move_m": AGENT_MOVEMENT_CONSTANT,
                "agent_rotation_deg": AGENT_ROTATION_DEG,
                "nav_graph_file": f"nav_graph-{self.collector.scene_name}.json",
                "notes": (
                    "This file is room/door survey layout only. Fine-grained "
                    "AI2-THOR reachability nodes/edges are in nav_graph-*.json "
                    "(GetReachablePositions), independent of the SPOC trajectory."
                ),
            },
        }

    def _export_nav_graph_snapshot(self, snapshot: str) -> None:
        """Export GetReachablePositions graph for the current scene state."""
        if snapshot == "episode_start":
            positions = self.reachable_positions
        else:
            positions = self.controller.get_reachable_positions()
            self.reachable_positions_end = positions
        door_states = self.get_door_states(force=True) or []
        graph = build_nav_graph_from_reachable_positions(
            positions,
            agent_move_m=AGENT_MOVEMENT_CONSTANT,
            agent_rotation_deg=AGENT_ROTATION_DEG,
            snapshot=snapshot,
            scene_id=self.collector.scene_name,
            episode_id=self.collector.episode_id,
            door_states=door_states,
        )
        self.collector.set_nav_graph(graph, snapshot=snapshot)

    def get_door_states(self, force: bool = False):
        """Return open/closed state for doors. Cached between interval steps."""
        t = self.collector.timestep
        if (
            not force
            and self._last_door_states is not None
            and t % self.door_log_interval != 0
        ):
            return None  # skip logging this step

        door_states = []
        for door in self.house.get("doors", []):
            door_id = door["id"]
            room0 = door.get("room0")
            room1 = door.get("room1")
            openable = door.get("openable", False)
            is_open = None
            openness = door.get("openness", None)
            position = (None, None, None)
            distance = None
            # Only query sim for openable doors (static doors don't change)
            if openable:
                try:
                    obj = self.controller.get_object(door_id)
                    openable = bool(obj.get("openable", openable))
                    is_open = bool(obj.get("isOpen", False)) if openable else None
                    openness = obj.get("openness", openness)
                    if openness is not None:
                        openness = float(np.round(openness, 4))
                    pos = obj.get("position")
                    if pos is not None:
                        position = self.collector.round_number(pos, 2)
                    if obj.get("distance") is not None:
                        distance = float(np.round(obj["distance"], 4))
                except Exception:
                    if openness is not None:
                        is_open = bool(openness > 0) if openable else None
            else:
                is_open = bool(openness > 0) if openness is not None else None
            door_states.append(
                {
                    "door_id": door_id,
                    "room0": room0,
                    "room1": room1,
                    "openable": openable,
                    "is_open": is_open,
                    "openness": openness,
                    "position": position,
                    "distance": distance,
                }
            )
        self._last_door_states = door_states
        return door_states

    def _ensure_pickupable_ids(self) -> Set[str]:
        """Cache pickupable object ids once (full metadata scan is expensive)."""
        if self._pickupable_ids is not None:
            return self._pickupable_ids
        ids = set()
        with self.controller.include_object_metadata_context():
            for o in self.controller.controller.last_event.metadata["objects"]:
                if o.get("pickupable", False):
                    ids.add(o["objectId"])
        self._pickupable_ids = ids
        return ids

    def _get_held_obj_id(self) -> Optional[str]:
        try:
            held = self.controller.get_held_objects()
            if held:
                return held[0]
        except Exception:
            pass
        return None

    def _gather_fov_all_objects(self, detections) -> List[Any]:
        """Metadata for named non-structural objects with pixels in the nav camera.

        Used for navigation-*.csv and objects-*.csv (spatial-relation candidates).
        Excludes walls/floors/ceilings/rooms and numeric-only ids (e.g. ``2|4``).
        Agent–room relations use navigation ``current-room`` instead.
        """
        result = []
        if detections is None:
            return result
        fov_ids = list(detections.keys())
        if not fov_ids:
            return result
        with self.controller.include_object_metadata_context():
            by_id = {
                o["objectId"]: o
                for o in self.controller.controller.last_event.metadata["objects"]
            }
        for oid in fov_ids:
            if not is_exportable_object(object_id=oid):
                continue
            obj = None
            if oid in by_id:
                obj = by_id[oid]
            else:
                try:
                    obj = self.controller.get_object(oid)
                except Exception:
                    continue
            if obj is None:
                continue
            if not is_exportable_object(obj_type=obj.get("objectType"), object_id=oid):
                continue
            result.append(obj)
        return result

    def _ensure_instance_colors(self, event) -> Dict[str, Any]:
        """Populate collector instance colors from an event (needed for mask pixels)."""
        if self.collector.dict_colors:
            return self.collector.dict_colors
        colors_meta = (event.metadata or {}).get("colors") or []
        self.collector.dict_colors = {
            d["name"]: d["color"] for d in colors_meta if "name" in d and "color" in d
        }
        return self.collector.dict_colors

    def _gather_pickupables_in_image(self, event) -> Dict[str, Any]:
        """Pickupables with mask pixels in the nav camera (matches ``images/`` / nav CSV).

        Uses ``instance_detections2D`` + segmentation ``visible-pixels > 0``, not THOR
        metadata ``visible``. Only valid when ``event`` ran ``renderImageSynthesis``.
        """
        detections = getattr(event, "instance_detections2D", None) or {}
        seg = getattr(event, "instance_segmentation_frame", None)
        if not detections or seg is None:
            return {}

        pickupable_ids = self._ensure_pickupable_ids()
        colors = self._ensure_instance_colors(event)
        result = {}
        for oid, bbox in detections.items():
            if oid not in pickupable_ids:
                continue
            color = colors.get(oid)
            if color is not None:
                try:
                    pixels = self.collector.get_visible_pixels_from_bbox(
                        event, bbox, color
                    )
                except Exception:
                    pixels = 0
                if pixels <= 0:
                    continue
            # No color yet: detection alone is enough to count as in-image.
            try:
                result[oid] = self.controller.get_object(
                    oid, include_receptacle_info=True
                )
            except Exception:
                with self.controller.include_object_metadata_context():
                    for o in self.controller.controller.last_event.metadata["objects"]:
                        if o.get("objectId") == oid:
                            result[oid] = o
                            break
        return result

    def _gather_visible_exportable_objects(self) -> List[Any]:
        """Named non-structural objects with THOR ``visible=True`` (every step).

        Restores fridge / door / window / furniture that older always-on-synthesis
        runs logged via FOV, without requiring ``renderImageSynthesis``.
        Pickupables already covered by tracking are included here too when visible.
        """
        result = []
        with self.controller.include_object_metadata_context():
            for o in self.controller.controller.last_event.metadata["objects"]:
                oid = o.get("objectId")
                if not o.get("visible", False):
                    continue
                if not is_exportable_object(
                    obj_type=o.get("objectType"), object_id=oid
                ):
                    continue
                try:
                    result.append(
                        self.controller.get_object(oid, include_receptacle_info=True)
                    )
                except Exception:
                    result.append(o)
        return result

    def _nav_synthesis_pass(self):
        """One Pass + instance synthesis (expensive). Prefer reusing an existing event."""
        return self.controller.controller.step(
            action="Pass", renderImageSynthesis=True
        )

    def _oid_has_mask_pixels(self, object_id: str, event) -> bool:
        """True if ``object_id`` has mask pixels on an event that already has synthesis."""
        detections = getattr(event, "instance_detections2D", None) or {}
        if object_id not in detections:
            return False
        seg = getattr(event, "instance_segmentation_frame", None)
        if seg is None:
            return True
        colors = self._ensure_instance_colors(event)
        color = colors.get(object_id)
        if color is None:
            return True
        try:
            return (
                self.collector.get_visible_pixels_from_bbox(
                    event, detections[object_id], color
                )
                > 0
            )
        except Exception:
            return True

    def _object_in_nav_image(self, object_id: str, event=None) -> bool:
        """Mask-pixel visibility. Reuses ``event`` when provided (no extra Pass)."""
        if event is None:
            try:
                event = self._nav_synthesis_pass()
            except Exception:
                return False
        return self._oid_has_mask_pixels(object_id, event)

    def _any_oid_in_nav_image(self, object_ids, event=None) -> bool:
        """One synthesis (if needed), then check several object ids."""
        ids = [oid for oid in object_ids if oid]
        if not ids:
            return False
        if event is None:
            try:
                event = self._nav_synthesis_pass()
            except Exception:
                return False
        return any(self._oid_has_mask_pixels(oid, event) for oid in ids)

    def _object_thor_visible(self, object_id: str) -> bool:
        """Cheap mid-loop check from metadata (no synthesis)."""
        try:
            obj = self.controller.get_object(object_id)
            return bool(obj.get("visible", False))
        except Exception:
            return False

    def _receptacles_in_room(self, room_id: str) -> List[Dict[str, Any]]:
        """Non-pickupable receptacles in ``room_id`` — **Floor excluded** from the pool."""
        if room_id in self._receptacles_by_room:
            return self._receptacles_by_room[room_id]
        receptacles = []
        with self.controller.include_object_metadata_context():
            for o in self.controller.controller.last_event.metadata["objects"]:
                if not o.get("receptacle", False):
                    continue
                if o.get("pickupable", False):
                    continue
                oid = o.get("objectId")
                # Floor is not a distinctive landmark for directional answers
                if is_structural_object(
                    obj_type=o.get("objectType"), object_id=oid
                ):
                    continue
                if oid is not None and str(oid).startswith("Floor|"):
                    continue
                if o.get("objectType") == "Floor":
                    continue
                try:
                    r_id, _ = self.controller.get_objects_room_id_and_type(oid)
                except Exception:
                    continue
                if r_id == room_id:
                    receptacles.append(o)
        self._receptacles_by_room[room_id] = receptacles
        return receptacles

    def _receptacle_is_usable(self, rec: Dict[str, Any]) -> bool:
        if rec.get("openable", False) and not rec.get("isOpen", False):
            return False
        return True

    def _receptacle_center_xz(self, rec: Dict[str, Any]):
        pos = rec.get("position")
        if pos is None:
            box = rec.get("axisAlignedBoundingBox") or {}
            pos = box.get("center")
        if pos is None:
            return None
        if isinstance(pos, dict):
            return float(pos["x"]), float(pos["z"])
        return float(pos[0]), float(pos[2])

    def _receptacle_salience(self, rec: Dict[str, Any]) -> float:
        """Rough visual prominence from AABB volume (larger = more salient decoy)."""
        box = rec.get("axisAlignedBoundingBox") or {}
        size = box.get("size")
        if not size:
            return 0.0
        if isinstance(size, dict):
            return abs(float(size.get("x", 0))) * abs(float(size.get("y", 0))) * abs(
                float(size.get("z", 0))
            )
        return abs(float(size[0])) * abs(float(size[1])) * abs(float(size[2]))

    def _restore_object_pose(self, object_id: str, position, rotation=None) -> bool:
        """Put object back after a rejected (still-visible) relocation."""
        if isinstance(position, (tuple, list)):
            position = {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        kwargs = dict(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position=position,
            forceKinematic=True,
        )
        if rotation is not None:
            if isinstance(rotation, (tuple, list)):
                rotation = {
                    "x": float(rotation[0]),
                    "y": float(rotation[1]),
                    "z": float(rotation[2]),
                }
            kwargs["rotation"] = rotation
        event = self.controller.controller.step(**kwargs)
        return bool(event.metadata.get("lastActionSuccess", False))

    def _xz_dist(self, p0, p1) -> float:
        if isinstance(p0, dict):
            x0, z0 = p0["x"], p0["z"]
        else:
            x0, z0 = p0[0], p0[2]
        if isinstance(p1, dict):
            x1, z1 = p1["x"], p1["z"]
        else:
            x1, z1 = p1[0], p1[2]
        return float(np.hypot(x1 - x0, z1 - z0))

    def _positions_close(self, p0, p1, tol: float = 0.2) -> bool:
        """True if world positions agree within ``tol`` meters (L2)."""
        if p0 is None or p1 is None:
            return False

        def _xyz(p):
            if isinstance(p, dict):
                return float(p["x"]), float(p["y"]), float(p["z"])
            return float(p[0]), float(p[1]), float(p[2])

        a, b = _xyz(p0), _xyz(p1)
        return float(np.linalg.norm(np.array(a) - np.array(b))) <= tol

    def _kinematic_place_on_receptacle(
        self,
        object_id: str,
        receptacle_id: str,
        ref_pos,
    ):
        """``PlaceObjectAtPoint`` + ``forceKinematic`` on a receptacle spawn point.

        Same placement mode as the real displace (deterministic kinematic place).
        Does **not** check visibility or restore. Returns (success, info).
        """
        info = {
            "receptacle_id": receptacle_id,
            "n_coords": 0,
            "n_place_attempts": 0,
            "last_error": None,
            "spawn_error": None,
            "placed_pos": None,
        }
        try:
            coords = self.controller.get_locations_on_receptacle(receptacle_id)
        except Exception as e:
            info["spawn_error"] = str(e)
            return False, info
        if not coords:
            info["spawn_error"] = "empty_spawn_coords"
            return False, info

        scored = []
        for pos in coords:
            d = self._xz_dist(ref_pos, pos)
            if d >= self.min_displace_distance:
                scored.append((d, pos))
        if not scored:
            # For distractor trials, still allow any spawn if none are far enough
            scored = [(self._xz_dist(ref_pos, pos), pos) for pos in coords]
        random.shuffle(scored)
        scored.sort(key=lambda t: -t[0])
        info["n_coords"] = len(scored)

        for _, pos in scored[: self.max_place_coords]:
            info["n_place_attempts"] += 1
            event = self.controller.controller.step(
                action="PlaceObjectAtPoint",
                objectId=object_id,
                position=pos,
                forceKinematic=True,
            )
            if not event.metadata.get("lastActionSuccess", False):
                err = event.metadata.get("errorMessage") or event.metadata.get(
                    "lastAction"
                )
                info["last_error"] = (
                    str(err) if err is not None else "PlaceObjectAtPoint_failed"
                )
                continue
            info["placed_pos"] = pos
            return True, info

        return False, info

    def _try_hidden_place_on_receptacle(
        self,
        object_id: str,
        receptacle_id: str,
        from_pos,
        from_rotation=None,
    ):
        """Place object on receptacle; mid-loop undoes with cheap THOR ``visible``.

        Authoritative mask-pixel check happens once after place succeeds (caller
        runs a single synthesis Pass). Avoids Pass+synthesis on every spawn try.
        Returns (success, info).
        """
        info = {
            "receptacle_id": receptacle_id,
            "n_coords": 0,
            "n_place_attempts": 0,
            "n_undone_visible": 0,
            "last_error": None,
            "spawn_error": None,
            "placed_pos": None,
        }
        try:
            coords = self.controller.get_locations_on_receptacle(receptacle_id)
        except Exception as e:
            info["spawn_error"] = str(e)
            return False, info
        if not coords:
            info["spawn_error"] = "empty_spawn_coords"
            return False, info

        scored = []
        for pos in coords:
            d = self._xz_dist(from_pos, pos)
            if d >= self.min_displace_distance:
                scored.append((d, pos))
        if not scored:
            info["spawn_error"] = "no_far_enough_spawn_coords"
            return False, info
        random.shuffle(scored)
        scored.sort(key=lambda t: -t[0])
        info["n_coords"] = len(scored)

        for _, pos in scored[: self.max_place_coords]:
            info["n_place_attempts"] += 1
            event = self.controller.controller.step(
                action="PlaceObjectAtPoint",
                objectId=object_id,
                position=pos,
                forceKinematic=True,
            )
            if not event.metadata.get("lastActionSuccess", False):
                err = event.metadata.get("errorMessage") or event.metadata.get(
                    "lastAction"
                )
                info["last_error"] = (
                    str(err) if err is not None else "PlaceObjectAtPoint_failed"
                )
                continue

            # Cheap filter only — final mask check is a single Pass in the caller
            if self._object_thor_visible(object_id):
                info["n_undone_visible"] += 1
                self._restore_object_pose(object_id, from_pos, from_rotation)
                info["last_error"] = "placed_but_thor_visible_undone"
                continue

            info["placed_pos"] = pos
            return True, info

        return False, info

    def _trial_place_readback(
        self,
        object_id: str,
        receptacle_id: str,
        ref_pos,
        restore_pos,
        restore_rot=None,
    ):
        """Kinematic trial place for a distractor; restore ``restore_pos`` after.

        Returns (ok, resolved_pos_rounded_or_None, receptacle_id_or_None).
        """
        ok, place_info = self._kinematic_place_on_receptacle(
            object_id, receptacle_id, ref_pos
        )
        if not ok:
            self._restore_object_pose(object_id, restore_pos, restore_rot)
            return False, None, None
        try:
            obj = self.controller.get_object(object_id, include_receptacle_info=True)
            resolved = self.collector.round_number(obj["position"], 2)
            parents = obj.get("parentReceptacles") or []
            parent = parents[0] if parents else receptacle_id
        except Exception:
            resolved = self.collector.round_number(
                place_info.get("placed_pos") or restore_pos, 2
            )
            parent = receptacle_id
        self._restore_object_pose(object_id, restore_pos, restore_rot)
        return True, resolved, parent

    def _pick_distractor_receptacles(
        self,
        receptacles: List[Dict[str, Any]],
        chosen_receptacle_id: str,
        chosen_pos,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Select nearby + salient-far receptacles (Floor already excluded from pool)."""
        chosen_center = None
        for rec in receptacles:
            if rec["objectId"] == chosen_receptacle_id:
                chosen_center = self._receptacle_center_xz(rec)
                break
        if chosen_center is None and chosen_pos is not None:
            if isinstance(chosen_pos, dict):
                chosen_center = (float(chosen_pos["x"]), float(chosen_pos["z"]))
            else:
                chosen_center = (float(chosen_pos[0]), float(chosen_pos[2]))

        nearby_candidates = []
        far_candidates = []
        for rec in receptacles:
            rid = rec["objectId"]
            if rid == chosen_receptacle_id:
                continue
            if not self._receptacle_is_usable(rec):
                continue
            center = self._receptacle_center_xz(rec)
            if center is None or chosen_center is None:
                continue
            d = float(np.hypot(center[0] - chosen_center[0], center[1] - chosen_center[1]))
            if d <= NEARBY_RECEPTACLE_XZ_M:
                nearby_candidates.append((d, rec))
            else:
                far_candidates.append((self._receptacle_salience(rec), d, rec))

        nearby_candidates.sort(key=lambda t: t[0])
        far_candidates.sort(key=lambda t: (-t[0], -t[1]))

        return {
            "nearby_receptacle": nearby_candidates[0][1] if nearby_candidates else None,
            "salient_decoy_location": far_candidates[0][2] if far_candidates else None,
        }

    def _kinematic_place_at_pose(self, object_id: str, position, rotation=None):
        """Place at an exact pose (used for object swap). Returns (ok, error_or_None)."""
        if isinstance(position, (tuple, list)):
            position = {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        kwargs = dict(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position=position,
            forceKinematic=True,
        )
        if rotation is not None:
            if isinstance(rotation, (tuple, list)):
                rotation = {
                    "x": float(rotation[0]),
                    "y": float(rotation[1]),
                    "z": float(rotation[2]),
                }
            kwargs["rotation"] = rotation
        event = self.controller.controller.step(**kwargs)
        ok = bool(event.metadata.get("lastActionSuccess", False))
        err = None
        if not ok:
            err = event.metadata.get("errorMessage") or event.metadata.get("lastAction")
            err = str(err) if err is not None else "PlaceObjectAtPoint_failed"
        return ok, err

    def _xyz_dict(self, p) -> Dict[str, float]:
        if isinstance(p, dict):
            return {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}

    def _floor_ids_in_room(self, room_id: str) -> List[str]:
        """Floor receptacle ids in ``room_id`` (swap park only — not destination sampling)."""
        ids = []
        with self.controller.include_object_metadata_context():
            for o in self.controller.controller.last_event.metadata["objects"]:
                oid = o.get("objectId")
                is_floor = o.get("objectType") == "Floor" or (
                    oid is not None and str(oid).startswith("Floor|")
                )
                if not is_floor:
                    continue
                try:
                    r_id, _ = self.controller.get_objects_room_id_and_type(oid)
                except Exception:
                    continue
                if r_id == room_id:
                    ids.append(oid)
        return ids

    def _floor_park_poses(self, pos_a, pos_b, room_id: str, max_n: int = 6) -> List[Dict[str, float]]:
        """Floor spawn points near the A–B midpoint (temporary hold, not a persisted destination)."""
        ax, az = self._xyz_dict(pos_a)["x"], self._xyz_dict(pos_a)["z"]
        bx, bz = self._xyz_dict(pos_b)["x"], self._xyz_dict(pos_b)["z"]
        mx, mz = (ax + bx) / 2.0, (az + bz) / 2.0
        scored = []
        for fid in self._floor_ids_in_room(room_id):
            try:
                coords = self.controller.get_locations_on_receptacle(fid)
            except Exception:
                continue
            if not coords:
                continue
            for pos in coords:
                p = self._xyz_dict(pos)
                d = float(np.hypot(p["x"] - mx, p["z"] - mz))
                scored.append((d, p))
        scored.sort(key=lambda t: t[0])
        return [p for _, p in scored[:max_n]]

    def _object_between_xz(
        self, pos_a, pos_b, skip_ids: Set[str], corridor_m: float = 0.4
    ) -> bool:
        """True if some other object sits on the xz segment between A and B."""
        a = self._xyz_dict(pos_a)
        b = self._xyz_dict(pos_b)
        abx, abz = b["x"] - a["x"], b["z"] - a["z"]
        ab2 = abx * abx + abz * abz
        if ab2 < 1e-6:
            return False
        with self.controller.include_object_metadata_context():
            objects = list(self.controller.controller.last_event.metadata["objects"])
        for o in objects:
            oid = o.get("objectId")
            if oid in skip_ids:
                continue
            if o.get("objectType") in ("Floor", "Wall", "Ceiling", "Room"):
                continue
            pos = o.get("position")
            if not pos:
                continue
            p = self._xyz_dict(pos)
            t = ((p["x"] - a["x"]) * abx + (p["z"] - a["z"]) * abz) / ab2
            if t <= 0.08 or t >= 0.92:
                continue
            cx = a["x"] + t * abx
            cz = a["z"] + t * abz
            if float(np.hypot(p["x"] - cx, p["z"] - cz)) <= corridor_m:
                return True
        return False

    def _swap_temp_park_poses(
        self, pos_a, pos_b, room_id: str, skip_ids: Set[str]
    ) -> List[Dict[str, float]]:
        """Holds so A can leave its pose before B moves in.

        Mid-air points plus **Floor** spawn coords. Floor is tried first when another
        object lies between A and B (mid-air would collide with it).
        """
        a = self._xyz_dict(pos_a)
        b = self._xyz_dict(pos_b)
        mid_y = max(a["y"], b["y"]) + 0.5
        midair = [
            {"x": (a["x"] + b["x"]) / 2.0, "y": mid_y, "z": (a["z"] + b["z"]) / 2.0},
            {"x": a["x"], "y": a["y"] + 0.5, "z": a["z"]},
            {"x": b["x"], "y": b["y"] + 0.5, "z": b["z"]},
            {"x": a["x"] + 0.35, "y": mid_y, "z": a["z"] + 0.35},
            {"x": b["x"] - 0.35, "y": mid_y, "z": b["z"] - 0.35},
        ]
        floor_poses = self._floor_park_poses(pos_a, pos_b, room_id)
        if self._object_between_xz(pos_a, pos_b, skip_ids):
            return floor_poses + midair
        return midair + floor_poses

    def _pick_swap_partner(
        self,
        primary_oid: str,
        primary_type: Optional[str],
        room_id: str,
        in_image_ids: Set[str],
    ) -> Optional[str]:
        """Another eligible hidden pickupable of a different type in the same room."""
        remaining = self.collector.max_displacements - len(
            self.collector.displaced_object_ids
        )
        if remaining < 2:
            return None
        for oid in self.collector.eligible_for_displacement():
            if oid == primary_oid:
                continue
            if oid in in_image_ids:
                continue
            track = self.collector.tracked_objects.get(oid) or {}
            other_type = track.get("obj_type")
            if primary_type and other_type and other_type == primary_type:
                continue
            try:
                other_room, _ = self.controller.get_objects_room_id_and_type(oid)
            except Exception:
                continue
            if other_room != room_id:
                continue
            return oid
        return None

    def _candidate_rows_after_place(
        self,
        object_id: str,
        from_pos,
        to_receptacle,
        to_pos_rounded,
        resolved_pos,
        after_rot,
        receptacles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Chosen + trial-teleport distractor rows for one persisted move."""
        rows = [
            {
                "event_id": None,
                "obj_id": object_id,
                "at_timestep": self.collector.timestep,
                "candidate_role": "chosen",
                "candidate_receptacle": to_receptacle,
                "candidate_pos": to_pos_rounded,
                "is_persisted": True,
            }
        ]
        distractors = self._pick_distractor_receptacles(
            receptacles, to_receptacle, resolved_pos
        )
        for role, rec in distractors.items():
            if rec is None:
                continue
            rid = rec["objectId"]
            ok_trial, trial_pos, trial_parent = self._trial_place_readback(
                object_id,
                rid,
                from_pos,
                restore_pos=resolved_pos,
                restore_rot=after_rot,
            )
            if not ok_trial or trial_pos is None:
                continue
            rows.append(
                {
                    "event_id": None,
                    "obj_id": object_id,
                    "at_timestep": self.collector.timestep,
                    "candidate_role": role,
                    "candidate_receptacle": trial_parent or rid,
                    "candidate_pos": trial_pos,
                    "is_persisted": False,
                }
            )
        return rows

    def _try_hidden_object_swap(
        self,
        oid_a: str,
        obj_a: Dict[str, Any],
        room_id: str,
        in_image_ids: Set[str],
        receptacles: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Fallback: swap two different-type hidden pickupables (both out of image).

        Returns two linked event dicts on success, else None (scene restored).
        """
        type_a = obj_a.get("objectType")
        oid_b = self._pick_swap_partner(oid_a, type_a, room_id, in_image_ids)
        if oid_b is None:
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_no_partner",
                    "detail": "no different-type hidden partner in room",
                    "room_id": room_id,
                }
            )
            return None

        try:
            obj_b = self.controller.get_object(oid_b, include_receptacle_info=True)
        except Exception as e:
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_get_partner",
                    "detail": str(e),
                    "room_id": room_id,
                    "to_receptacle": oid_b,
                }
            )
            return None

        if oid_b in in_image_ids:
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_partner_in_image",
                    "detail": f"partner {oid_b} still in nav image",
                    "room_id": room_id,
                }
            )
            return None

        parents_a = obj_a.get("parentReceptacles") or []
        parents_b = obj_b.get("parentReceptacles") or []
        rec_a = parents_a[0] if parents_a else None
        rec_b = parents_b[0] if parents_b else None
        pos_a = obj_a["position"]
        pos_b = obj_b["position"]
        rot_a = obj_a.get("rotation")
        rot_b = obj_b.get("rotation")
        from_a = self.collector.round_number(pos_a, 2)
        from_b = self.collector.round_number(pos_b, 2)
        vis_a = bool(obj_a.get("visible", False))
        vis_b = bool(obj_b.get("visible", False))

        # Three-step swap: park A (mid-air or Floor), move B into A's pose, move A into B's pose.
        # Floor is preferred when another object sits between A and B.
        parked = False
        park_err = None
        park_poses = self._swap_temp_park_poses(
            pos_a, pos_b, room_id, skip_ids={oid_a, oid_b}
        )
        for temp in park_poses:
            ok, park_err = self._kinematic_place_at_pose(oid_a, temp)
            if ok:
                parked = True
                break
        if not parked:
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_park_a",
                    "detail": f"failed parking {oid_a} before swap; err={park_err}",
                    "room_id": room_id,
                    "from_receptacle": rec_a,
                    "to_receptacle": rec_b,
                    "last_error": park_err or "swap_park_a_failed",
                }
            )
            return None

        ok_b, err_b = self._kinematic_place_at_pose(oid_b, pos_a, rot_a)
        if not ok_b:
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_place_b",
                    "detail": f"failed placing {oid_b} at {oid_a} pose; err={err_b}",
                    "room_id": room_id,
                    "from_receptacle": rec_a,
                    "to_receptacle": rec_b,
                    "last_error": err_b or "swap_place_b_failed",
                }
            )
            return None

        ok_a, err_a = self._kinematic_place_at_pose(oid_a, pos_b, rot_b)
        if not ok_a:
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self._restore_object_pose(oid_b, pos_b, rot_b)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_place_a",
                    "detail": f"failed placing {oid_a} at partner pose; err={err_a}",
                    "room_id": room_id,
                    "from_receptacle": rec_a,
                    "to_receptacle": rec_b,
                    "last_error": err_a or "swap_place_a_failed",
                }
            )
            return None

        if self._any_oid_in_nav_image([oid_a, oid_b]):
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self._restore_object_pose(oid_b, pos_b, rot_b)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_in_image_after",
                    "detail": f"undone swap {oid_a}<->{oid_b}; still in nav image",
                    "room_id": room_id,
                    "last_error": "swap_still_in_image",
                }
            )
            return None

        try:
            after_a = self.controller.get_object(oid_a, include_receptacle_info=True)
            after_b = self.controller.get_object(oid_b, include_receptacle_info=True)
        except Exception as e:
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self._restore_object_pose(oid_b, pos_b, rot_b)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_get_after",
                    "detail": str(e),
                    "room_id": room_id,
                }
            )
            return None

        to_a = self.collector.round_number(after_a["position"], 2)
        to_b = self.collector.round_number(after_b["position"], 2)
        if not self._positions_close(after_a["position"], pos_b, tol=0.35):
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self._restore_object_pose(oid_b, pos_b, rot_b)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_state_mismatch",
                    "detail": f"{oid_a} not at partner pose after swap",
                    "room_id": room_id,
                    "last_error": "swap_position_mismatch",
                }
            )
            return None
        if not self._positions_close(after_b["position"], pos_a, tol=0.35):
            self._restore_object_pose(oid_a, pos_a, rot_a)
            self._restore_object_pose(oid_b, pos_b, rot_b)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid_a,
                    "status": "fail",
                    "stage": "swap_state_mismatch",
                    "detail": f"{oid_b} not at partner pose after swap",
                    "room_id": room_id,
                    "last_error": "swap_position_mismatch",
                }
            )
            return None

        parents_after_a = after_a.get("parentReceptacles") or []
        parents_after_b = after_b.get("parentReceptacles") or []
        to_rec_a = parents_after_a[0] if parents_after_a else rec_b
        to_rec_b = parents_after_b[0] if parents_after_b else rec_a

        event_id = f"disp_{len(self.collector.data_displacement_events)}"
        t = self.collector.timestep

        cand_a = self._candidate_rows_after_place(
            oid_a,
            from_pos=pos_a,
            to_receptacle=to_rec_a,
            to_pos_rounded=to_a,
            resolved_pos=after_a["position"],
            after_rot=after_a.get("rotation"),
            receptacles=receptacles,
        )
        cand_b = self._candidate_rows_after_place(
            oid_b,
            from_pos=pos_b,
            to_receptacle=to_rec_b,
            to_pos_rounded=to_b,
            resolved_pos=after_b["position"],
            after_rot=after_b.get("rotation"),
            receptacles=receptacles,
        )

        event_a = {
            "event_id": event_id,
            "obj_id": oid_a,
            "at_timestep": t,
            "action": "PlaceObjectAtPoint",
            "from_receptacle": rec_a,
            "to_receptacle": to_rec_a,
            "from_pos": from_a,
            "to_pos": to_a,
            "hidden_during": True,
            "visible_just_before": vis_a,
            "visible_just_after": bool(after_a.get("visible", False)),
            "in_fov_just_before": oid_a in in_image_ids,
            "in_fov_just_after": False,
            "moved_via": "swap",
            "swap_partner_id": oid_b,
            "notes": "object_swap",
        }
        event_b = {
            "event_id": event_id,
            "obj_id": oid_b,
            "at_timestep": t,
            "action": "PlaceObjectAtPoint",
            "from_receptacle": rec_b,
            "to_receptacle": to_rec_b,
            "from_pos": from_b,
            "to_pos": to_b,
            "hidden_during": True,
            "visible_just_before": vis_b,
            "visible_just_after": bool(after_b.get("visible", False)),
            "in_fov_just_before": oid_b in in_image_ids,
            "in_fov_just_after": False,
            "moved_via": "swap",
            "swap_partner_id": oid_a,
            "notes": "object_swap",
        }

        self.collector.log_displacement_event(event_a)
        self.collector.log_displacement_event(event_b)
        for row in cand_a + cand_b:
            row["event_id"] = event_id
            self.collector.log_displacement_candidate(row)
        self.collector.log_displacement_debug(
            {
                "obj_id": oid_a,
                "status": "ok",
                "stage": "object_swap",
                "detail": (
                    f"swapped with {oid_b} "
                    f"n_candidates={len(cand_a) + len(cand_b)}"
                ),
                "room_id": room_id,
                "from_receptacle": rec_a,
                "to_receptacle": to_rec_a,
                "last_error": None,
            }
        )
        return [event_a, event_b]


    def maybe_displace_hidden_objects(
        self, in_image_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Move tracked pickupables only while they are absent from the nav image.

        Seen = mask pixels in the agent camera at least once; displace after ≥2
        synthesis steps without mask pixels; keep the move only if still not in
        the image after place. If receptacle place fails, try an **object swap**
        with a different-type hidden partner (both out of image).

        ``in_image_ids`` comes from the **same** nav-step synthesis event used for
        tracking/CSV (no second Pass for eligibility). Mid-place undoes use cheap
        THOR ``visible``; one synthesis Pass runs only for the final mask check
        (and one Pass for swap post-check covering both objects).
        If ``in_image_ids`` is ``None``, skip displacement (cannot verify image FOV).
        """
        events = []
        if in_image_ids is None:
            return events
        if self._displacements_this_step >= self.max_displacements_per_step:
            return events
        if len(self.collector.displaced_object_ids) >= self.collector.max_displacements:
            return events

        candidates = self.collector.candidates_for_displacement()
        if not candidates:
            return events

        # Only attempt one candidate per step (max_displacements_per_step)
        candidates = candidates[:1]

        for oid in candidates:
            if self._displacements_this_step >= self.max_displacements_per_step:
                break

            # Hard requirement: not in the agent image before moving
            if oid in in_image_ids:
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "still_in_image",
                        "detail": "refusing displace while object has nav mask pixels",
                    }
                )
                continue

            try:
                obj_before = self.controller.get_object(oid, include_receptacle_info=True)
            except Exception as e:
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "get_object",
                        "detail": str(e),
                    }
                )
                continue
            if not obj_before.get("pickupable", False):
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "not_pickupable",
                        "detail": "metadata pickupable=False",
                    }
                )
                continue

            room_id, _ = self.controller.get_objects_room_id_and_type(oid)
            if room_id is None:
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "no_room_id",
                        "detail": "get_objects_room_id_and_type returned None",
                    }
                )
                continue

            parents_before = obj_before.get("parentReceptacles") or []
            from_receptacle = parents_before[0] if parents_before else None
            from_pos = obj_before["position"]
            from_pos_rounded = self.collector.round_number(from_pos, 2)
            from_rot = obj_before.get("rotation")
            visible_before = bool(obj_before.get("visible", False))
            in_image_before = oid in in_image_ids

            receptacles = list(self._receptacles_in_room(room_id))
            # Prefer current receptacle first (cup left→right on same table), then others
            if from_receptacle is not None:
                receptacles.sort(
                    key=lambda r: 0 if r["objectId"] == from_receptacle else 1
                )
            else:
                random.shuffle(receptacles)

            n_closed = 0
            n_spawn_empty = 0
            n_place_fail = 0
            n_undone_visible = 0
            n_tried = 0
            last_error = None
            placed = False
            to_receptacle = None
            to_pos = None
            for rec in receptacles:
                if n_tried >= self.max_receptacles_to_try:
                    break
                rid = rec["objectId"]
                if not self._receptacle_is_usable(rec):
                    n_closed += 1
                    continue
                n_tried += 1
                ok, place_info = self._try_hidden_place_on_receptacle(
                    oid, rid, from_pos, from_rot
                )
                n_undone_visible += int(place_info.get("n_undone_visible") or 0)
                if place_info.get("spawn_error"):
                    n_spawn_empty += 1
                    last_error = place_info["spawn_error"]
                elif not ok:
                    n_place_fail += 1
                    last_error = place_info.get("last_error")
                if ok:
                    placed = True
                    to_receptacle = rid
                    to_pos = place_info.get("placed_pos")
                    break

            if not placed:
                self.collector.note_place_failure(oid)
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "place_failed",
                        "detail": (
                            f"no hidden PlaceObjectAtPoint; "
                            f"last_error={last_error}; trying object_swap"
                        ),
                        "room_id": room_id,
                        "from_receptacle": from_receptacle,
                        "to_receptacle": None,
                        "n_receptacles_room": len(receptacles),
                        "n_receptacles_tried": n_tried,
                        "n_skipped_same_parent": 0,
                        "n_skipped_closed": n_closed,
                        "n_spawn_empty": n_spawn_empty,
                        "n_place_fail": n_place_fail,
                        "last_error": last_error,
                    }
                )
                swap_events = self._try_hidden_object_swap(
                    oid, obj_before, room_id, in_image_ids, receptacles
                )
                if swap_events:
                    events.extend(swap_events)
                    self._displacements_this_step += 1
                continue

            try:
                obj_after = self.controller.get_object(oid, include_receptacle_info=True)
            except Exception as e:
                self._restore_object_pose(oid, from_pos, from_rot)
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "get_object_after",
                        "detail": str(e),
                        "room_id": room_id,
                        "from_receptacle": from_receptacle,
                        "to_receptacle": to_receptacle,
                    }
                )
                continue

            in_image_after = self._object_in_nav_image(oid)
            if in_image_after:
                self._restore_object_pose(oid, from_pos, from_rot)
                self.collector.note_place_failure(oid)
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "in_image_after_final_check",
                        "detail": "undone; object still has nav mask pixels",
                        "room_id": room_id,
                        "from_receptacle": from_receptacle,
                        "to_receptacle": to_receptacle,
                        "n_undone_visible": n_undone_visible + 1,
                        "last_error": "final_in_image_check_failed",
                    }
                )
                continue

            visible_after = bool(obj_after.get("visible", False))
            resolved_pos = obj_after["position"]
            to_pos_rounded = self.collector.round_number(resolved_pos, 2)
            parents_after = obj_after.get("parentReceptacles") or []
            if to_receptacle is None and parents_after:
                to_receptacle = parents_after[0]

            # Validate engine state matches the intended place before persisting
            expected_pos = to_pos if to_pos is not None else resolved_pos
            pos_ok = self._positions_close(resolved_pos, expected_pos, tol=0.25)
            parent_ok = True
            if to_receptacle is not None and parents_after:
                parent_ok = to_receptacle in parents_after or parents_after[0] == to_receptacle
            # Reject Floor dropouts even when xz matches a counter spawn
            if parents_after and any(
                str(p).startswith("Floor|") or p == "Floor" for p in parents_after
            ):
                parent_ok = False
            if not pos_ok or not parent_ok:
                self._restore_object_pose(oid, from_pos, from_rot)
                self.collector.note_place_failure(oid)
                mismatch = []
                if not pos_ok:
                    mismatch.append(
                        f"pos expected={self.collector.round_number(expected_pos, 2)} "
                        f"got={to_pos_rounded}"
                    )
                if not parent_ok:
                    mismatch.append(
                        f"parent expected={to_receptacle} got={parents_after}"
                    )
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "state_mismatch",
                        "detail": "readback mismatch; " + "; ".join(mismatch),
                        "room_id": room_id,
                        "from_receptacle": from_receptacle,
                        "to_receptacle": to_receptacle,
                        "n_place_fail": n_place_fail + 1,
                        "last_error": "object_state_mismatch",
                    }
                )
                continue

            after_rot = obj_after.get("rotation")

            # Trial-teleport distractors (same PlaceObjectAtPoint mode); restore to real pose
            distractors = self._pick_distractor_receptacles(
                receptacles, to_receptacle, resolved_pos
            )
            candidate_rows = [
                {
                    "event_id": None,  # filled after event_id assigned
                    "obj_id": oid,
                    "at_timestep": self.collector.timestep,
                    "candidate_role": "chosen",
                    "candidate_receptacle": to_receptacle,
                    "candidate_pos": to_pos_rounded,
                    "is_persisted": True,
                }
            ]
            for role, rec in distractors.items():
                if rec is None:
                    continue
                rid = rec["objectId"]
                ok_trial, trial_pos, trial_parent = self._trial_place_readback(
                    oid,
                    rid,
                    from_pos,
                    restore_pos=resolved_pos,
                    restore_rot=after_rot,
                )
                if not ok_trial or trial_pos is None:
                    continue
                candidate_rows.append(
                    {
                        "event_id": None,
                        "obj_id": oid,
                        "at_timestep": self.collector.timestep,
                        "candidate_role": role,
                        "candidate_receptacle": trial_parent or rid,
                        "candidate_pos": trial_pos,
                        "is_persisted": False,
                    }
                )

            same_rec = from_receptacle is not None and from_receptacle == to_receptacle
            notes = (
                "same_receptacle_hidden_shift"
                if same_rec
                else "other_receptacle_hidden_place"
            )

            event_id = f"disp_{len(self.collector.data_displacement_events)}"
            event = {
                "event_id": event_id,
                "obj_id": oid,
                "at_timestep": self.collector.timestep,
                "action": "PlaceObjectAtPoint",
                "from_receptacle": from_receptacle,
                "to_receptacle": to_receptacle,
                "from_pos": from_pos_rounded,
                "to_pos": to_pos_rounded,
                "hidden_during": True,
                "visible_just_before": visible_before,
                "visible_just_after": visible_after,
                # Schema: image FOV (nav mask), not THOR metadata visible
                "in_fov_just_before": in_image_before,
                "in_fov_just_after": False,
                "moved_via": "direct",
                "swap_partner_id": None,
                "notes": notes,
            }
            self.collector.log_displacement_event(event)
            for row in candidate_rows:
                row["event_id"] = event_id
                self.collector.log_displacement_candidate(row)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid,
                    "status": "ok",
                    "stage": "placed_hidden",
                    "detail": (
                        f"notes={notes} n_undone_visible={n_undone_visible} "
                        f"parent_ok={parent_ok} n_candidates={len(candidate_rows)}"
                    ),
                    "room_id": room_id,
                    "from_receptacle": from_receptacle,
                    "to_receptacle": to_receptacle,
                    "n_receptacles_room": len(receptacles),
                    "n_receptacles_tried": n_tried,
                    "n_skipped_same_parent": 0,
                    "n_skipped_closed": n_closed,
                    "n_spawn_empty": n_spawn_empty,
                    "n_place_fail": n_place_fail,
                    "last_error": None,
                }
            )
            events.append(event)
            self._displacements_this_step += 1
        return events

    def _build_object_state_rows(self, detections=None) -> List[Dict[str, Any]]:
        """Per-timestep true state for all tracked objects, including hidden.

        ``visible`` is THOR metadata; ``in_camera_fov`` is nav-image presence
        (mask pixels), updated on synthesis strides and used for displacement.
        Optional ``detections`` is ignored (kept for call compat).
        """
        tracked_ids = set(self.collector.tracked_objects.keys())
        if not tracked_ids:
            return []
        rows = []
        receptacle_open = {}
        for oid in tracked_ids:
            try:
                obj = self.controller.get_object(oid, include_receptacle_info=True)
            except Exception:
                continue
            parents = obj.get("parentReceptacles") or []
            rec_open = None
            if parents:
                pid = parents[0]
                if pid not in receptacle_open:
                    try:
                        parent = self.controller.get_object(pid)
                        if parent.get("openable", False):
                            receptacle_open[pid] = bool(parent.get("isOpen", False))
                        else:
                            receptacle_open[pid] = None
                    except Exception:
                        receptacle_open[pid] = None
                rec_open = receptacle_open[pid]
            track = self.collector.tracked_objects.get(oid) or {}
            in_camera_fov = bool(track.get("in_camera_fov", False))
            rows.append(
                {
                    "obj_meta": obj,
                    "in_camera_fov": in_camera_fov,
                    "receptacle_is_open": rec_open,
                }
            )
        return rows

    def _gather_tracked_object_meta(self) -> List[Any]:
        """Metadata for tracked pickupables (no FOV / detections required)."""
        result = []
        for oid in self.collector.tracked_objects:
            try:
                result.append(
                    self.controller.get_object(oid, include_receptacle_info=True)
                )
            except Exception:
                continue
        return result

    def min_l2_distance_to_target(self):
        distances = self.get_room_distances()
        if len(distances) > 0:
            return min(distances)
        else:
            return 0

    def min_geodesic_distance_to_target(self):
        return -1

    def get_agent_loc(self):
        agent_position = self.controller.get_current_agent_position()
        return round(agent_position["x"], 1), round(agent_position["z"], 1)

    def get_room_distances(self):
        agent_position = self.controller.get_current_agent_position()
        p = Point(agent_position["x"], agent_position["z"])
        distances = []
        for r, m in self.room_poly_map.items():
            if r not in self.seen_rooms:
                dis = m.distance(p)
                if dis > 0:
                    distances.append(dis)
        return distances

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names[action]
        self.last_taken_action_str = action_str

        self._took_sub_done_action = False
        self._displacements_this_step = 0

        # Eval patches task.max_steps after __init__; keep collector + FOV stride in sync
        self._sync_collector_horizon()

        if action_str == THORActions.done:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
            self._export_nav_graph_snapshot("episode_end")
            self.collector.save_data(reason="done")
        elif action_str == THORActions.sub_done:
            self.num_sub_done += 1
            self._took_sub_done_action = True
            if self.previous_room not in self.seen_rooms:
                self.num_successful_sub_done += 1
                self.last_action_success = True
                self.seen_rooms.append(self.previous_room)
                self.closest_distance = self.dist_to_target_func()
            else:
                self.last_action_success = False
        else:
            # Expensive: instance_detections2D / masks / segmentation frame.
            # Gate on collector.timestep so logged frames stay evenly spaced.
            render_mask_this_step = self.collector.timestep % self.stride == 0
            event = self.controller.agent_step(
                action=action_str,
                render_image_synthesis=render_mask_this_step,
            )
            self.last_action_success = bool(event)

            if not self.collector.at_capacity:
                current_room = self.get_current_room()
                room_info = {
                    "current_room": current_room,
                    "current_room_type": (
                        self.room_type_dict.get(current_room)
                        if current_room is not None
                        else None
                    ),
                    "seen_rooms": list(self.seen_rooms),
                }
                door_states = self.get_door_states()

                # Navigation CSV: only objects with mask pixels in this frame
                # (same signal as displacement image FOV). Hidden pickupables stay
                # in object_state / displacement_* — not padded into navigation.
                by_id = {}
                include_detections = False
                if render_mask_this_step:
                    detections = event.instance_detections2D
                    if (
                        event.instance_segmentation_frame is not None
                        and detections is not None
                    ):
                        include_detections = True
                        export_dets = self.collector.filter_export_detections(
                            detections or {}
                        )
                        for o in self._gather_fov_all_objects(export_dets):
                            by_id[o["objectId"]] = o
                objects = list(by_id.values())

                # Displacement tracking uses nav-image mask pixels (same as images/),
                # only on synthesis strides. Off-stride: freeze hidden_steps.
                in_image_pickupables = {}
                if include_detections:
                    in_image_pickupables = self._gather_pickupables_in_image(event)
                    self.collector.update_visibility_tracking(in_image_pickupables)
                    self.maybe_displace_hidden_objects(
                        in_image_ids=set(in_image_pickupables.keys())
                    )

                object_states = self._build_object_state_rows()

                self.collector.collect_data(
                    event,
                    action_str,
                    objects,
                    self.controller.controller,
                    room_info=room_info,
                    door_states=door_states,
                    action_success=self.last_action_success,
                    held_obj_id=self._get_held_obj_id(),
                    object_states=object_states,
                    include_detections=include_detections,
                )

            position = self.controller.get_current_agent_position()
            self.path.append(position)

            if len(self.path) > 1:
                self.travelled_distance += position_dist(
                    p0=self.path[-1], p1=self.path[-2], ignore_y=True
                )

        # Horizon end (no agent `done`): still export CSVs up to max_steps
        if (
            not self._took_end_action
            and self.num_steps_taken() + 1 >= self.max_steps
        ):
            self._export_nav_graph_snapshot("episode_end")
            self.collector.save_data(reason="max_steps")

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def successful_if_done(self, percentage_seen=None, strict_success=False) -> bool:
        return len(self.seen_rooms) == len(self.house["rooms"])

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        return self.reward_shaper.shaping()

    def judge(self) -> float:
        if self.reward_config is None:
            return 0
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = dict(
            coverage=len(self.seen_rooms) / len(self.house["rooms"]),
            distance=self.travelled_distance,
            ep_length=self.num_steps_taken(),
            total_reward=np.sum(self._rewards),
            num_seen_rooms=len(self.seen_rooms),
            num_visited_rooms=len(self.visited_rooms),
            num_visited_locations=len(self.visited_loc),
            success=self._success,
            num_sub_done=self.num_sub_done,
            sub_done_acc=(
                self.num_successful_sub_done / self.num_sub_done if self.num_sub_done > 0 else 0.0
            ),
            num_displacements=len(self.collector.data_displacement_events),
        )
        self._metrics = metrics
        return metrics
