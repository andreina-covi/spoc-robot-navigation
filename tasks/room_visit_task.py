from typing import Any, Dict, List, Optional, Set

import numpy as np
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from shapely.geometry import Point, Polygon
from typing_extensions import Literal

from environment.stretch_controller import StretchController
from tasks.abstract_task import AbstractSPOCTask
from utils.data_generation_utils.navigation_utils import get_room_id_from_location
from utils.distance_calculation_utils import position_dist
from utils.type_utils import RewardConfig, THORActions
from utils.constants.object_constants import is_exportable_object, is_structural_object
from utils.object_labels import describe_object
from utils.constants.stretch_initialization_utils import (
    AGENT_MOVEMENT_CONSTANT,
    AGENT_ROTATION_DEG,
)
from utils.mask_stride import mask_render_stride
from utils.nav_graph_export import (
    build_nav_graph_from_reachable_positions,
    reachability_grid_size,
)
from utils.placement_anchors import salient_landmarks
from training.online.reward.reward_shaper import RoomVisitRewardShaper
from collector import Collector
from online_evaluation.max_episode_configs import MAX_EPISODE_LEN_PER_TASK

ROOM_VISIT_HORIZON = MAX_EPISODE_LEN_PER_TASK["RoomVisit"]


def room_id_at(room_poly_map, position):
    """Room that contains this xz position. Missing or empty positions stay unset."""
    if not position:
        return None
    return get_room_id_from_location(room_poly_map, position, verbose=False)


def room_polygon_and_area(room):
    points = room.get("floorPolygon") or []
    polygon = [
        {"x": point["x"], "y": point.get("y", 0.0), "z": point["z"]} for point in points
    ]
    area = 0.0
    if len(points) >= 3:
        area = float(Polygon([(point["x"], point["z"]) for point in points]).area)
    return polygon, round(area, 4)

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
            max_steps=collector_max_steps,
            flush_every=50,
        )
        self.collector.house_index = house_index if house_index != "unknown" else None
        self.collector.house_split = self.task_info.get("house_split")
        self.collector.set_world_layout(self.build_world_layout())
        self._export_nav_graph_snapshot("episode_start")
        self._pickupable_ids: Optional[Set[str]] = None
        self.stride = mask_render_stride(collector_max_steps, ROOM_VISIT_HORIZON)
        self._sync_collector_horizon()

    def _sync_collector_horizon(self):
        """Keep collector max_steps / FOV stride / nav camera params in sync."""
        if self.max_steps is not None and self.max_steps > 0:
            self.collector.max_steps = self.max_steps
        self.stride = mask_render_stride(self.max_steps, ROOM_VISIT_HORIZON)
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
            polygon, area = room_polygon_and_area(room)
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
                    "polygon": polygon,
                    "area": area,
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

        # Landmarks: top objects per room whose box sits on the floor.
        # Objaverse instances qualify the same way; type names are not used.
        raw_objects = []
        with self.controller.include_object_metadata_context():
            raw_objects = [
                dict(obj) for obj in self.controller.controller.last_event.metadata["objects"]
            ]
        for obj in raw_objects:
            obj["room_id"] = room_id_at(self.room_poly_map, obj.get("position"))
        landmarks = []
        for obj in salient_landmarks(raw_objects):
            oid = obj.get("objectId") or obj.get("id")
            label = describe_object(oid, obj.get("objectType"), obj.get("assetId"))
            pos = obj.get("position", {"x": 0, "y": 0, "z": 0})
            room_id = obj.get("room_id")
            landmarks.append(
                {
                    "landmark_id": oid,
                    "obj-type": label.obj_type,
                    "obj-synset": label.synset,
                    "position": pos,
                    "region_id": room_id,
                }
            )
            if room_id is not None:
                for r in regions:
                    if r["region_id"] == room_id:
                        r["landmark_obj_ids"].append(oid)
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
        if snapshot == "episode_start" or not self.collector.initial_door_states:
            door_states = self.get_door_states() or []
            self.collector.record_initial_doors(door_states)
        else:
            door_states = self.collector.initial_door_states
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

    def get_door_states(self):
        """Open/closed state for every house door."""
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
        """Metadata for non-structural objects with pixels in the nav camera.

        Numeric ids such as ``6|5`` are included. Walls, floors, ceilings, and
        rooms are not. Each object gets ``roomId`` for the objects CSV.
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
            obj["roomId"] = room_id_at(self.room_poly_map, obj.get("position"))
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

    def _build_object_state_rows(self, detections=None) -> List[Dict[str, Any]]:
        """Per-timestep true state for all tracked objects, including hidden.

        ``visible`` is THOR metadata; ``in_camera_fov`` is nav-image presence
        (mask pixels) on steps that rendered a mask.
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
            # Instance masks render on every agent step.
            render_mask_this_step = self.collector.timestep % self.stride == 0
            event = self.controller.agent_step(
                action=action_str,
                render_image_synthesis=render_mask_this_step,
            )
            self.last_action_success = bool(event.metadata.get("lastActionSuccess", False))

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

                # Navigation CSV: only objects with mask pixels in this frame.
                # Hidden pickupables stay in object_state, not in navigation.
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

                # Mask pixels are recorded on every step.
                if include_detections:
                    in_image_pickupables = self._gather_pickupables_in_image(event)
                    self.collector.update_visibility_tracking(in_image_pickupables)

                object_states = self._build_object_state_rows()

                self.collector.collect_data(
                    event,
                    action_str,
                    objects,
                    self.controller.controller,
                    room_info=room_info,
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
        )
        self._metrics = metrics
        return metrics
