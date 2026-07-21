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
from utils.constants.object_constants import is_exportable_object
from training.online.reward.reward_shaper import RoomVisitRewardShaper
from collector import Collector

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
            max_displacements=5,
            max_steps=collector_max_steps,
            flush_every=50,
        )
        self.collector.set_world_layout(self.build_world_layout())
        self._displacements_this_step = 0
        self.max_displacements_per_step = 1
        self.max_receptacles_to_try = 4
        self.max_place_coords = 4
        self.min_displace_distance = 0.25  # meters; same table left↔right is OK
        self.door_log_interval = 5  # log doors every N steps (not every step)
        self._pickupable_ids: Optional[Set[str]] = None
        self._receptacles_by_room: Dict[str, List[Dict[str, Any]]] = {}
        self._last_door_states = None

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
                "grid_size": None,
                "notes": "Regions from ProcTHOR rooms; passages from house doors.",
            },
        }

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
        Agent–room relations use current-room / region_trajectory instead.
        """
        if detections is None:
            return []
        result = []
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

    def _gather_fov_pickupable_meta(self, detections) -> Dict[str, Any]:
        """Pickupable objects in nav FOV — used only for displacement tracking."""
        if detections is None:
            return {}
        pickupable_ids = self._ensure_pickupable_ids()
        result = {}
        for oid in detections.keys():
            if oid not in pickupable_ids:
                continue
            try:
                result[oid] = self.controller.get_object(oid, include_receptacle_info=True)
            except Exception:
                continue
        return result

    def _receptacles_in_room(self, room_id: str) -> List[Dict[str, Any]]:
        if room_id in self._receptacles_by_room:
            return self._receptacles_by_room[room_id]
        receptacles = []
        with self.controller.include_object_metadata_context():
            for o in self.controller.controller.last_event.metadata["objects"]:
                if not o.get("receptacle", False):
                    continue
                if o.get("pickupable", False):
                    continue
                try:
                    r_id, _ = self.controller.get_objects_room_id_and_type(o["objectId"])
                except Exception:
                    continue
                if r_id == room_id:
                    receptacles.append(o)
        self._receptacles_by_room[room_id] = receptacles
        return receptacles

    def _refresh_nav_detections(self):
        self.controller.controller.step("Pass", renderImageSynthesis=True)
        return self.controller.controller.last_event.instance_detections2D or {}

    def _oid_in_nav_fov(self, object_id: str, detections=None) -> bool:
        if detections is None:
            detections = self.controller.controller.last_event.instance_detections2D or {}
        return object_id in detections

    def _restore_object_pose(self, object_id: str, position, rotation=None) -> bool:
        """Put object back after a rejected (still-visible) relocation."""
        if isinstance(position, (tuple, list)):
            position = {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}
        kwargs = dict(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position=position,
            forceKinematic=True,
        )
        if rotation is not None:
            if isinstance(rotation, (tuple, list)):
                rotation = {"x": float(rotation[0]), "y": float(rotation[1]), "z": float(rotation[2])}
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

    def _try_hidden_place_on_receptacle(
        self,
        object_id: str,
        receptacle_id: str,
        from_pos,
        from_rotation=None,
    ):
        """Place object on receptacle only if the new pose stays outside nav FOV.

        If a candidate pose becomes visible in the nav camera, undo and try another.
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

        # Prefer points far enough from the original pose (e.g. other side of table)
        scored = []
        for pos in coords:
            d = self._xz_dist(from_pos, pos)
            if d >= self.min_displace_distance:
                scored.append((d, pos))
        if not scored:
            info["spawn_error"] = "no_far_enough_spawn_coords"
            return False, info
        random.shuffle(scored)
        scored.sort(key=lambda t: -t[0])  # try farthest first
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
                err = event.metadata.get("errorMessage") or event.metadata.get("lastAction")
                info["last_error"] = str(err) if err is not None else "PlaceObjectAtPoint_failed"
                continue

            detections_after = self._refresh_nav_detections()
            if self._oid_in_nav_fov(object_id, detections_after):
                # Would "appear from nothing" in the saved nav image — reject & undo
                info["n_undone_visible"] += 1
                self._restore_object_pose(object_id, from_pos, from_rotation)
                self._refresh_nav_detections()
                info["last_error"] = "placed_but_visible_in_nav_fov_undone"
                continue

            info["placed_pos"] = pos
            return True, info

        return False, info

    def maybe_displace_hidden_objects(self, detections) -> List[Dict[str, Any]]:
        """Move previously seen pickupable objects only while they stay out of nav FOV.

        Goal: agent saw object at L0, later L0 is off-camera, object moves to L1 while
        still off-camera — so it never pops into the navigation view mid-episode.
        """
        events = []
        if self._displacements_this_step >= self.max_displacements_per_step:
            return events
        if len(self.collector.displaced_object_ids) >= self.collector.max_displacements:
            return events

        candidates = self.collector.candidates_for_displacement(detections)
        if not candidates:
            return events

        # Only attempt one candidate per step (max_displacements_per_step)
        candidates = candidates[:1]

        fov_ids = set(detections.keys()) if detections is not None else set()
        for oid in candidates:
            if self._displacements_this_step >= self.max_displacements_per_step:
                break

            # Hard requirement: not in nav camera before moving
            if oid in fov_ids:
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "still_in_fov",
                        "detail": "refusing displace while object is in nav detections",
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
            in_fov_before = False

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
                if rec.get("openable", False) and not rec.get("isOpen", False):
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
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "place_failed",
                        "detail": (
                            f"no hidden PlaceObjectAtPoint; "
                            f"last_error={last_error}"
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

            detections_after = (
                self.controller.controller.last_event.instance_detections2D or {}
            )
            in_fov_after = oid in detections_after
            if in_fov_after:
                # Final safety: never keep a pop-into-view relocation
                self._restore_object_pose(oid, from_pos, from_rot)
                self._refresh_nav_detections()
                self.collector.log_displacement_debug(
                    {
                        "obj_id": oid,
                        "status": "fail",
                        "stage": "visible_after_final_check",
                        "detail": "undone; object was still in nav FOV",
                        "room_id": room_id,
                        "from_receptacle": from_receptacle,
                        "to_receptacle": to_receptacle,
                        "n_undone_visible": n_undone_visible + 1,
                        "last_error": "final_fov_check_failed",
                    }
                )
                continue

            to_pos_rounded = self.collector.round_number(
                to_pos if to_pos is not None else obj_after["position"], 2
            )
            parents_after = obj_after.get("parentReceptacles") or []
            if to_receptacle is None and parents_after:
                to_receptacle = parents_after[0]

            same_rec = from_receptacle is not None and from_receptacle == to_receptacle
            notes = (
                "same_receptacle_hidden_shift"
                if same_rec
                else "other_receptacle_hidden_place"
            )

            event = {
                "event_id": f"disp_{len(self.collector.data_displacement_events)}",
                "obj_id": oid,
                "at_timestep": self.collector.timestep,
                "action": "PlaceObjectAtPoint",
                "from_receptacle": from_receptacle,
                "to_receptacle": to_receptacle,
                "from_pos": from_pos_rounded,
                "to_pos": to_pos_rounded,
                "hidden_during": True,
                "visible_just_before": visible_before,
                "visible_just_after": bool(obj_after.get("visible", False)),
                "in_fov_just_before": in_fov_before,
                "in_fov_just_after": False,
                "moved_via": "direct",
                "notes": notes,
            }
            self.collector.log_displacement_event(event)
            self.collector.log_displacement_debug(
                {
                    "obj_id": oid,
                    "status": "ok",
                    "stage": "placed_hidden",
                    "detail": f"notes={notes} n_undone_visible={n_undone_visible}",
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

    def _build_object_state_rows(self, detections) -> List[Dict[str, Any]]:
        """Per-timestep true state for all tracked (interest) objects, including hidden."""
        tracked_ids = set(self.collector.tracked_objects.keys())
        if not tracked_ids:
            return []
        fov_ids = set(detections.keys()) if detections is not None else set()
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
            rows.append(
                {
                    "obj_meta": obj,
                    "in_camera_fov": oid in fov_ids,
                    "receptacle_is_open": rec_open,
                }
            )
        return rows

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

        # Eval patches task.max_steps after __init__; keep collector in sync
        if self.max_steps is not None and self.max_steps > 0:
            self.collector.max_steps = self.max_steps

        if action_str == THORActions.done:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
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
            event = self.controller.agent_step(action=action_str)
            self.last_action_success = bool(event)
            # Skip expensive FOV logging once the collector step budget is full
            if not self.collector.at_capacity:
                detections = self.controller.controller.last_event.instance_detections2D
                if (
                    self.controller.controller.last_event.instance_segmentation_frame is None
                    or detections is None
                ):
                    print("instance segmentation is none")
                else:
                    raw_detections = (
                        self.controller.controller.last_event.instance_detections2D or {}
                    )
                    # Export: named non-structural FOV objects (+ visibility metrics in CSV)
                    export_dets = self.collector.filter_export_detections(raw_detections)
                    objects = self._gather_fov_all_objects(export_dets)
                    objects_by_id = {o["objectId"]: o for o in objects}

                    # Displacement "seen": only recognizable (shown % / size thresholds)
                    recognizable = self.collector.filter_recognizable_detections(
                        export_dets,
                        self.controller.controller,
                        objects_by_id=objects_by_id,
                    )

                    # 1) Discover pickupables only when recognizable; "hidden" uses raw FOV
                    pickupable_meta = self._gather_fov_pickupable_meta(recognizable)
                    self.collector.update_visibility_tracking(raw_detections, pickupable_meta)

                    # 2) Relocate when out of ALL nav pixels
                    self.maybe_displace_hidden_objects(raw_detections)

                    # 3) Log named non-structural FOV objects; post-process uses metrics
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
                    object_states = self._build_object_state_rows(raw_detections)
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
