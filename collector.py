import os
import json
import datetime

import numpy as np
import pandas as pd
import cv2

from utils.constants.objaverse_data_dirs import OBJAVERSE_NAVIGATION_PATH
from utils.constants.stretch_initialization_utils import (
    AGENT_MOVEMENT_CONSTANT,
    AGENT_ROTATION_DEG,
    ARM_MOVE_CONSTANT,
    HORIZON,
    INTEL_CAMERA_WIDTH,
    INTEL_CAMERA_HEIGHT,
    INTEL_VERTICAL_FOV,
    WRIST_ROTATION,
)
from utils.constants.object_constants import is_exportable_object
from utils.type_utils import THORActions


class Collector:
    """Logs navigation + cm-benchmark fields for invisible_displacement / survey.

    Hard-caps logging at ``max_steps`` (like an LLM context window): once that many
    timesteps are recorded, further ``collect_data`` calls are no-ops. Step buffers
    are flushed to CSV periodically so long episodes do not keep the full table in RAM.

    Episode layout under ``OBJAVERSE_NAVIGATION_PATH/<timestamp>/``:
    - ``images/`` — RGB frames
    - ``annotations/`` — all CSV and JSON exports
    """

    # Default visibility thresholds (396×224 nav frame).
    # Tuned so a human can typically recognize the object, not a 1–2 px peek.
    DEFAULT_MIN_FRAME_FRACTION = 0.009  # ~0.90% of the frame
    FRAME_WIDTH = INTEL_CAMERA_WIDTH
    FRAME_HEIGHT = INTEL_CAMERA_HEIGHT
    FRAME_SIZE_PX = FRAME_WIDTH * FRAME_HEIGHT
    VERTICAL_FOV_DEG = INTEL_VERTICAL_FOV
    # Minimum visible area derived from frame coverage (used as bbox-area threshold in fast mode)
    MIN_MASK_PIXELS = FRAME_SIZE_PX * DEFAULT_MIN_FRAME_FRACTION
    DEFAULT_MIN_BBOX_SIDE = 12  # drops thin edge slivers
    # Strict mode only: visible_mask / projected_silhouette
    DEFAULT_MIN_UNOCCLUDED_RATIO = 0.4
    # "fast" = bbox-only (default, collection-safe); "strict" = masks + 3D silhouette (slow)
    DEFAULT_VISIBILITY_MODE = "fast"

    # Step-growing CSVs flushed incrementally; small / end-of-episode files written in save_data.
    _STREAM_TABLES = (
        "navigation",
        "doors",
        "object_state",
        "region_trajectory",
        "passage_state",
        "displacement_debug",
    )

    def __init__(
        self,
        scene_name=None,
        episode_kind="invisible_displacement",
        environment="ai2thor",
        max_displacements=5,
        max_steps=None,
        flush_every=50,
        min_frame_fraction=None,
        min_unoccluded_ratio=None,
        min_mask_pixels=None,
        min_bbox_side=None,
        visibility_mode=None,
    ):
        self.dict_agent = {}
        self.data_objects = {"objects": set()}
        self.data_doors = []
        self.data_object_state = []
        self.data_displacement_events = []
        self.data_displacement_debug = []  # attempts / failures for diagnosing 0 displacements
        self.data_region_trajectory = []
        self.timestep = 0
        # ProcTHOR always reports sceneName="Procedural"; use house_index instead
        self.scene_name = scene_name if scene_name is not None else "unknown"
        self.episode_kind = episode_kind
        self.environment = environment
        self.dt = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        self.episode_id = f"{self.scene_name}_{self.dt}"
        # Episode root: two sibling folders — images/ (frames) and annotations/ (CSV + JSON)
        self.out_dir = os.path.join(OBJAVERSE_NAVIGATION_PATH, self.dt)
        self.image_path = os.path.join(self.out_dir, "images")
        self.annotations_dir = os.path.join(self.out_dir, "annotations")
        self.dict_colors = {}
        self.visited_rooms = set()
        self.max_displacements = max_displacements
        # Context-window style cap: keep at most this many logged steps
        self.max_steps = max_steps
        self.flush_every = max(1, int(flush_every))
        self._saved = False
        self._csv_initialized = set()
        self._truncated = False
        self._save_reason = None

        self.visibility_mode = (
            self.DEFAULT_VISIBILITY_MODE
            if visibility_mode is None
            else str(visibility_mode).lower()
        )
        self.min_frame_fraction = (
            self.DEFAULT_MIN_FRAME_FRACTION
            if min_frame_fraction is None
            else float(min_frame_fraction)
        )
        # If the caller overrides frame fraction, recompute MIN_MASK_PIXELS-equivalent
        self.min_mask_pixels = (
            self.FRAME_SIZE_PX * self.min_frame_fraction
            if min_mask_pixels is None
            else float(min_mask_pixels)
        )
        self.min_bbox_side = (
            self.DEFAULT_MIN_BBOX_SIDE if min_bbox_side is None else float(min_bbox_side)
        )
        self.min_unoccluded_ratio = (
            self.DEFAULT_MIN_UNOCCLUDED_RATIO
            if min_unoccluded_ratio is None
            else float(min_unoccluded_ratio)
        )

        # Track objects seen in nav FOV for invisible displacement
        # oid -> {obj_type, first_seen_t, last_in_fov_t, hidden_steps, displaced}
        self.tracked_objects = {}
        self.displaced_object_ids = set()
        self.world_layout = None

        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

    @property
    def at_capacity(self) -> bool:
        """True when the step budget is exhausted (no more rows will be logged).

        ``max_steps <= 0`` means unset / unlimited (online eval often constructs
        the task with ``max_steps=-1`` before patching the real horizon).
        """
        return (
            self.max_steps is not None
            and self.max_steps > 0
            and self.timestep >= self.max_steps
        )

    def round_number(self, arr_numbers, n_round):
        if type(arr_numbers) is dict:
            rounded_arr = [arr_numbers["x"], arr_numbers["y"], arr_numbers["z"]]
        else:
            rounded_arr = arr_numbers
        return tuple([np.round(number, n_round) for number in rounded_arr])

    @staticmethod
    def bbox_width_height_area(bbox):
        """Return (width, height, area) for an instance_detections2D bbox [cmin,rmin,cmax,rmax]."""
        if bbox is None or len(bbox) < 4:
            return 0.0, 0.0, 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return width, height, width * height

    @staticmethod
    def get_mask_pixel_count(object_id, instance_masks):
        """Count visible pixels for ``object_id`` from THOR ``instance_masks``."""
        if not instance_masks or object_id not in instance_masks:
            return None
        mask = instance_masks[object_id]
        if mask is None:
            return None
        return int(np.count_nonzero(mask))

    def estimate_full_silhouette_from_size_distance(self, obj_meta, frame_hw=None):
        """Estimate full on-screen object area from 3D size and distance.

        Uses the two largest AABB/OBB extents:
        ``(f * a / dist) * (f * b / dist)``. This is the denominator for
        ``visible_area / full_object_area`` (shown fraction of the object).
        """
        if not obj_meta:
            return None
        dist = obj_meta.get("distance")
        if dist is None:
            return None
        dist = float(dist)
        if dist < 0.05:
            return None

        size = None
        for key in ("objectOrientedBoundingBox", "axisAlignedBoundingBox"):
            box = obj_meta.get(key)
            if box and box.get("size") is not None:
                size = box["size"]
                break
        if size is None:
            return None

        try:
            extents = sorted(
                [float(size["x"]), float(size["y"]), float(size["z"])], reverse=True
            )
        except Exception:
            return None
        a, b = extents[0], extents[1]
        if a <= 0 or b <= 0:
            return None

        if frame_hw is None:
            height = INTEL_CAMERA_HEIGHT
        else:
            height = int(frame_hw[0])
        fy = (height / 2.0) / np.tan(np.deg2rad(self.VERTICAL_FOV_DEG) / 2.0)
        area = float((fy * a / dist) * (fy * b / dist))
        return area if area > 1.0 else None

    def compute_visibility_metrics(
        self, bbox, mask_pixels=None, frame_hw=None, silhouette_pixels=None
    ):
        """How much of the object is visible in the nav frame (metrics only, no keep/drop)."""
        width, height, bbox_area = self.bbox_width_height_area(bbox)
        used_mask = mask_pixels is not None
        visible_area = float(mask_pixels) if used_mask else float(bbox_area)

        frame_px = float(self.FRAME_SIZE_PX)
        if frame_hw is not None:
            fh, fw = int(frame_hw[0]), int(frame_hw[1])
            frame_px = float(max(1, fh * fw))
        visible_frac = visible_area / frame_px

        unoccluded_ratio = None
        if silhouette_pixels is not None and silhouette_pixels > 0:
            unoccluded_ratio = min(1.0, visible_area / float(silhouette_pixels))

        return {
            "bbox_width": width,
            "bbox_height": height,
            "bbox_area": bbox_area,
            "visible_area_px": visible_area,
            "visible_frac": visible_frac,
            "silhouette_pixels": silhouette_pixels,
            "unoccluded_ratio": unoccluded_ratio,
            "used_mask": used_mask,
        }

    def passes_visibility_filters(self, metrics) -> bool:
        """Recognizability gate for invisible-displacement discovery (not written to CSV).

        Requires sufficient bbox area, min side length, frame fraction, and (when known)
        shown fraction of the object. Strict mode also checks mask pixel count.
        """
        if metrics["bbox_area"] < self.min_mask_pixels:
            return False
        if min(metrics["bbox_width"], metrics["bbox_height"]) < self.min_bbox_side:
            return False
        if metrics["visible_frac"] is not None and metrics["visible_frac"] < self.min_frame_fraction:
            return False
        if (
            metrics["unoccluded_ratio"] is not None
            and metrics["unoccluded_ratio"] < self.min_unoccluded_ratio
        ):
            return False
        if self.visibility_mode == "strict" and metrics["visible_area_px"] < self.min_mask_pixels:
            return False
        return True

    def filter_export_detections(self, detections):
        """Detections for CSV export: named, non-structural objects (any nav pixels).

        Visibility thresholds are **not** applied — metrics go to CSV for post-processing.
        Drops Wall/Floor/etc. and numeric-only ids like ``2|4``.
        """
        if not detections:
            return {}
        return {
            oid: bbox
            for oid, bbox in detections.items()
            if is_exportable_object(object_id=oid)
        }

    def filter_recognizable_detections(self, detections, controller, objects_by_id=None):
        """Keep detections that pass recognizability thresholds (displacement / "seen").

        Not used for navigation export. Cheap bbox checks first, then size/distance
        shown-fraction when object metadata is available.
        """
        if not detections:
            return {}
        event = controller.last_event
        frame = getattr(event, "frame", None)
        frame_hw = None if frame is None else frame.shape[:2]

        candidates = {}
        for oid, bbox in detections.items():
            if not is_exportable_object(object_id=oid):
                continue
            width, height, area = self.bbox_width_height_area(bbox)
            if area < self.min_mask_pixels or min(width, height) < self.min_bbox_side:
                continue
            candidates[oid] = bbox
        if not candidates:
            return {}

        if objects_by_id is None:
            objects_by_id = {}
            try:
                for o in event.metadata.get("objects", []):
                    oid = o.get("objectId")
                    if oid in candidates:
                        objects_by_id[oid] = o
            except Exception:
                objects_by_id = {}

        use_strict = self.visibility_mode == "strict"
        instance_masks = (getattr(event, "instance_masks", None) or {}) if use_strict else {}

        kept = {}
        for oid, bbox in candidates.items():
            obj_meta = objects_by_id.get(oid)
            if obj_meta is not None and not is_exportable_object(
                obj_type=obj_meta.get("objectType"), object_id=oid
            ):
                continue

            mask_px = (
                self.get_mask_pixel_count(oid, instance_masks) if use_strict else None
            )
            silhouette = None
            if obj_meta is not None:
                silhouette = self.estimate_full_silhouette_from_size_distance(
                    obj_meta, frame_hw=frame_hw
                )

            metrics = self.compute_visibility_metrics(
                bbox,
                mask_pixels=mask_px,
                frame_hw=frame_hw,
                silhouette_pixels=silhouette,
            )
            if self.passes_visibility_filters(metrics):
                kept[oid] = bbox
        return kept

    def get_object_data(self, arr_objects, controller, min_distance=0.0, max_distance=None):
        """Collect named non-structural FOV objects with visibility metrics.

        Does not drop for failing recognizability thresholds. Each nav row includes
        ``visible-area-px``, ``visible-frac``, ``unoccluded-ratio``, and
        ``full-silhouette-px`` so post-processing can decide. Skips structural meshes
        and numeric-only ids (e.g. ``2|4``).
        """
        objects = set()
        cond_objs = []
        detections = controller.last_event.instance_detections2D
        if detections is None:
            return cond_objs, objects

        event = controller.last_event
        frame = getattr(event, "frame", None)
        frame_hw = None if frame is None else frame.shape[:2]
        use_strict = self.visibility_mode == "strict"
        instance_masks = (
            (getattr(event, "instance_masks", None) or {}) if use_strict else {}
        )
        pos_dict_default = {"x": 0.0, "y": 0.0, "z": 0.0}

        for obj_dict in arr_objects:
            oid = obj_dict["objectId"]
            obj_type = obj_dict.get("objectType")
            if not is_exportable_object(obj_type=obj_type, object_id=oid):
                continue
            if oid not in detections:
                continue

            bbox = detections[oid]
            _, _, area = self.bbox_width_height_area(bbox)
            if area <= 0:
                continue

            mask_px = self.get_mask_pixel_count(oid, instance_masks) if use_strict else None
            silhouette = self.estimate_full_silhouette_from_size_distance(
                obj_dict, frame_hw=frame_hw
            )
            metrics = self.compute_visibility_metrics(
                bbox,
                mask_pixels=mask_px,
                frame_hw=frame_hw,
                silhouette_pixels=silhouette,
            )

            dist = float(obj_dict.get("distance") or 0.0)
            if dist < min_distance:
                continue
            if max_distance is not None and dist > max_distance:
                continue

            # [oid, dist, bbox, visible_area_px, visible_frac, unoccluded_ratio,
            #  full_silhouette_px]
            cond_objs.append(
                [
                    oid,
                    np.round(dist, 4),
                    bbox,
                    int(round(metrics["visible_area_px"])),
                    float(np.round(metrics["visible_frac"], 6)),
                    None
                    if metrics["unoccluded_ratio"] is None
                    else float(np.round(metrics["unoccluded_ratio"], 4)),
                    None
                    if metrics["silhouette_pixels"] is None
                    else float(np.round(metrics["silhouette_pixels"], 2)),
                ]
            )

            aabb = obj_dict.get("axisAlignedBoundingBox")
            oobb = obj_dict.get("objectOrientedBoundingBox")
            box = aabb if aabb is not None else oobb
            if box is None:
                box = {}

            color = self.dict_colors.get(oid)
            if color is None:
                continue
            objects.add(
                (
                    obj_type,
                    oid,
                    tuple(color),
                    tuple(self.round_number(obj_dict["position"], 2)),
                    tuple(self.round_number(obj_dict["rotation"], 2)),
                    tuple(obj_dict["receptacleObjectIds"])
                    if obj_dict.get("receptacleObjectIds") is not None
                    else (),
                    tuple(self.round_number(box.get("center", pos_dict_default), 2)),
                    tuple(self.round_number(box.get("size", pos_dict_default), 2)),
                )
            )
        return cond_objs, objects

    def save_data_by_axis(self, dict_data, base_name, array):
        axis_names = ["-x", "-y", "-z"]
        for axis, item in zip(axis_names, array):
            dict_data[base_name + axis].append(item)

    def save_bbox(self, dict_data, bbox):
        assert len(bbox) == 4, "size error of bbox, it must be of size 4"
        dict_data["cmin"].append(bbox[0])
        dict_data["rmin"].append(bbox[1])
        dict_data["cmax"].append(bbox[2])
        dict_data["rmax"].append(bbox[3])

    def add_basic_navigation_data(self, dict_navigation, key):
        dict_navigation["timestep"].append(key[0])
        dict_navigation["ag-action"].append(key[1])
        dict_navigation["action_success"].append(self.dict_agent[key]["action_success"])
        dict_navigation["held_obj-id"].append(self.dict_agent[key]["held_obj_id"])
        self.save_data_by_axis(dict_navigation, "ag-pos", self.dict_agent[key]["position"])
        self.save_data_by_axis(dict_navigation, "ag-rot", self.dict_agent[key]["rotation"])
        self.save_data_by_axis(dict_navigation, "camera-pos", self.dict_agent[key]["camera_position"])
        dict_navigation["degrees"].append(self.dict_agent[key]["degrees"])
        dict_navigation["camera-horizon"].append(self.dict_agent[key]["camera_horizon"])
        dict_navigation["path"].append(self.dict_agent[key]["image"])
        dict_navigation["current-room"].append(self.dict_agent[key]["current_room"])
        dict_navigation["current-room-type"].append(self.dict_agent[key]["current_room_type"])
        dict_navigation["room-just-entered"].append(self.dict_agent[key]["room_just_entered"])
        dict_navigation["visited-rooms"].append(self.dict_agent[key]["visited_rooms"])
        dict_navigation["seen-rooms"].append(self.dict_agent[key]["seen_rooms"])

    def save_data_navigation(self, dict_navigation, key):
        objects_data = self.dict_agent[key]["objects"]
        if not objects_data:
            self.add_basic_navigation_data(dict_navigation, key)
            dict_navigation["obj-id"].append(None)
            dict_navigation["obj-distance"].append(None)
            self.save_bbox(dict_navigation, [None, None, None, None])
            dict_navigation["visible-area-px"].append(None)
            dict_navigation["visible-frac"].append(None)
            dict_navigation["unoccluded-ratio"].append(None)
            dict_navigation["full-silhouette-px"].append(None)
            return

        for object_data in objects_data:
            self.add_basic_navigation_data(dict_navigation, key)
            dict_navigation["obj-id"].append(object_data[0])
            dict_navigation["obj-distance"].append(object_data[1])
            self.save_bbox(dict_navigation, object_data[2])
            if len(object_data) >= 7:
                dict_navigation["visible-area-px"].append(object_data[3])
                dict_navigation["visible-frac"].append(object_data[4])
                dict_navigation["unoccluded-ratio"].append(object_data[5])
                dict_navigation["full-silhouette-px"].append(object_data[6])
            else:
                dict_navigation["visible-area-px"].append(None)
                dict_navigation["visible-frac"].append(None)
                dict_navigation["unoccluded-ratio"].append(None)
                dict_navigation["full-silhouette-px"].append(None)

    def save_image(self, image_name, event):
        cv2.imwrite(image_name, event.cv2img)

    def get_dict_navigation(self):
        # episode/scene ids live in episode_meta.json (folder already identifies the run)
        dict_navigation = {
            "timestep": [],
            "ag-action": [],
            "action_success": [],
            "held_obj-id": [],
            "degrees": [],
            "ag-pos-x": [],
            "ag-pos-y": [],
            "ag-pos-z": [],
            "ag-rot-x": [],
            "ag-rot-y": [],
            "ag-rot-z": [],
            "cmin": [],
            "rmin": [],
            "cmax": [],
            "rmax": [],
            "camera-horizon": [],
            "camera-pos-x": [],
            "camera-pos-y": [],
            "camera-pos-z": [],
            "current-room": [],
            "current-room-type": [],
            "room-just-entered": [],
            "visited-rooms": [],
            "seen-rooms": [],
            "obj-id": [],
            "obj-distance": [],
            "visible-area-px": [],
            "visible-frac": [],
            "unoccluded-ratio": [],
            "full-silhouette-px": [],
            "path": [],
        }
        for key in self.dict_agent:
            self.save_data_navigation(dict_navigation, key)
        return dict_navigation

    def get_dict_objects(self):
        dict_objects = {
            "obj-type": [],
            "obj-id": [],
            "obj-color": [],
            "obj-pos-x": [],
            "obj-pos-y": [],
            "obj-pos-z": [],
            "obj-rot-x": [],
            "obj-rot-y": [],
            "obj-rot-z": [],
            "receptacleObjectIds": [],
            "bBox-center-x": [],
            "bBox-center-y": [],
            "bBox-center-z": [],
            "size-x": [],
            "size-y": [],
            "size-z": [],
        }
        for t in self.data_objects["objects"]:
            dict_objects["obj-type"].append(t[0])
            dict_objects["obj-id"].append(t[1])
            dict_objects["obj-color"].append(t[2])
            self.save_data_by_axis(dict_objects, "obj-pos", t[3])
            self.save_data_by_axis(dict_objects, "obj-rot", t[4])
            dict_objects["receptacleObjectIds"].append(t[5])
            self.save_data_by_axis(dict_objects, "bBox-center", t[6])
            self.save_data_by_axis(dict_objects, "size", t[7])
        return dict_objects

    def get_dict_doors(self):
        dict_doors = {
            "timestep": [],
            "door-id": [],
            "room0": [],
            "room1": [],
            "openable": [],
            "is-open": [],
            "openness": [],
            "door-pos-x": [],
            "door-pos-y": [],
            "door-pos-z": [],
            "distance": [],
        }
        for entry in self.data_doors:
            dict_doors["timestep"].append(entry["timestep"])
            dict_doors["door-id"].append(entry["door_id"])
            dict_doors["room0"].append(entry["room0"])
            dict_doors["room1"].append(entry["room1"])
            dict_doors["openable"].append(entry["openable"])
            dict_doors["is-open"].append(entry["is_open"])
            dict_doors["openness"].append(entry["openness"])
            self.save_data_by_axis(dict_doors, "door-pos", entry["position"])
            dict_doors["distance"].append(entry["distance"])
        return dict_doors

    def get_dict_object_state(self):
        cols = {
            "timestep": [],
            "obj-id": [],
            "obj-type": [],
            "obj-pos-x": [],
            "obj-pos-y": [],
            "obj-pos-z": [],
            "obj-rot-x": [],
            "obj-rot-y": [],
            "obj-rot-z": [],
            "visible": [],
            "in_camera_fov": [],
            "parent_receptacle": [],
            "parent_receptacles": [],
            "is_inside_receptacle": [],
            "receptacle_is_open": [],
            "distance_from_agent": [],
        }
        for entry in self.data_object_state:
            cols["timestep"].append(entry["timestep"])
            cols["obj-id"].append(entry["obj_id"])
            cols["obj-type"].append(entry["obj_type"])
            self.save_data_by_axis(cols, "obj-pos", entry["position"])
            self.save_data_by_axis(cols, "obj-rot", entry["rotation"])
            cols["visible"].append(entry["visible"])
            cols["in_camera_fov"].append(entry["in_camera_fov"])
            cols["parent_receptacle"].append(entry["parent_receptacle"])
            cols["parent_receptacles"].append(entry["parent_receptacles"])
            cols["is_inside_receptacle"].append(entry["is_inside_receptacle"])
            cols["receptacle_is_open"].append(entry["receptacle_is_open"])
            cols["distance_from_agent"].append(entry["distance_from_agent"])
        return cols

    def get_dict_displacement_events(self):
        cols = {
            "event_id": [],
            "obj-id": [],
            "at_timestep": [],
            "action": [],
            "from_receptacle": [],
            "to_receptacle": [],
            "from_pos-x": [],
            "from_pos-y": [],
            "from_pos-z": [],
            "to_pos-x": [],
            "to_pos-y": [],
            "to_pos-z": [],
            "hidden_during": [],
            "visible_just_before": [],
            "visible_just_after": [],
            "in_fov_just_before": [],
            "in_fov_just_after": [],
            "moved_via": [],
            "notes": [],
        }
        for entry in self.data_displacement_events:
            cols["event_id"].append(entry["event_id"])
            cols["obj-id"].append(entry["obj_id"])
            cols["at_timestep"].append(entry["at_timestep"])
            cols["action"].append(entry["action"])
            cols["from_receptacle"].append(entry["from_receptacle"])
            cols["to_receptacle"].append(entry["to_receptacle"])
            self.save_data_by_axis(cols, "from_pos", entry["from_pos"])
            self.save_data_by_axis(cols, "to_pos", entry["to_pos"])
            cols["hidden_during"].append(entry["hidden_during"])
            cols["visible_just_before"].append(entry["visible_just_before"])
            cols["visible_just_after"].append(entry["visible_just_after"])
            cols["in_fov_just_before"].append(entry["in_fov_just_before"])
            cols["in_fov_just_after"].append(entry["in_fov_just_after"])
            cols["moved_via"].append(entry["moved_via"])
            cols["notes"].append(entry["notes"])
        return cols

    def get_dict_passage_state(self):
        """Survey-oriented view of door open/closed state."""
        cols = {
            "timestep": [],
            "passage_id": [],
            "obj-id": [],
            "is_open": [],
            "is_locked": [],
            "from_region": [],
            "to_region": [],
        }
        for entry in self.data_doors:
            cols["timestep"].append(entry["timestep"])
            cols["passage_id"].append(entry["door_id"])
            cols["obj-id"].append(entry["door_id"])
            cols["is_open"].append(entry["is_open"])
            cols["is_locked"].append(None)
            cols["from_region"].append(entry["room0"])
            cols["to_region"].append(entry["room1"])
        return cols

    def get_dict_region_trajectory(self):
        cols = {
            "timestep": [],
            "region_id": [],
            "region_type": [],
        }
        for entry in self.data_region_trajectory:
            cols["timestep"].append(entry["timestep"])
            cols["region_id"].append(entry["region_id"])
            cols["region_type"].append(entry["region_type"])
        return cols

    def update_visibility_tracking(self, detections, pickupable_meta):
        """Update which pickupable objects have been seen / are currently hidden.

        pickupable_meta should only include objects currently in the nav FOV
        (and optionally already-tracked ids) to avoid full-scene scans each step.
        """
        fov_ids = set(detections.keys()) if detections is not None else set()
        # Update already-tracked objects
        for oid, track in self.tracked_objects.items():
            if oid in fov_ids:
                track["last_in_fov_t"] = self.timestep
                track["hidden_steps"] = 0
            else:
                track["hidden_steps"] += 1
        # Discover new pickupables only from FOV metadata
        for oid, meta in pickupable_meta.items():
            if oid in self.tracked_objects:
                continue
            if oid not in fov_ids:
                continue
            self.tracked_objects[oid] = {
                "obj_type": meta.get("objectType"),
                "first_seen_t": self.timestep,
                "last_in_fov_t": self.timestep,
                "hidden_steps": 0,
                "displaced": False,
            }

    def candidates_for_displacement(self, detections):
        """Pickupable objects seen before, out of nav FOV for >=1 step, not yet moved."""
        if len(self.displaced_object_ids) >= self.max_displacements:
            return []
        fov_ids = set(detections.keys()) if detections is not None else set()
        candidates = []
        for oid, track in self.tracked_objects.items():
            if track["displaced"] or oid in self.displaced_object_ids:
                continue
            if oid in fov_ids:
                continue
            # Need >=2 hidden steps so object_state can show hidden-at-L0 before the move
            if track["hidden_steps"] < 2:
                continue
            if track["first_seen_t"] >= self.timestep:
                continue
            candidates.append(oid)
        remaining = self.max_displacements - len(self.displaced_object_ids)
        return candidates[:remaining]

    def log_object_state_row(self, timestep, obj_meta, in_camera_fov, receptacle_is_open=None):
        parents = obj_meta.get("parentReceptacles") or []
        parent = parents[0] if parents else None
        pos = self.round_number(obj_meta["position"], 2)
        rot = self.round_number(obj_meta["rotation"], 2)
        dist = obj_meta.get("distance")
        self.data_object_state.append(
            {
                "timestep": timestep,
                "obj_id": obj_meta["objectId"],
                "obj_type": obj_meta["objectType"],
                "position": pos,
                "rotation": rot,
                "visible": bool(obj_meta.get("visible", False)),
                "in_camera_fov": bool(in_camera_fov),
                "parent_receptacle": parent,
                "parent_receptacles": json.dumps(parents),
                "is_inside_receptacle": parent is not None,
                "receptacle_is_open": receptacle_is_open,
                "distance_from_agent": None if dist is None else float(np.round(dist, 4)),
            }
        )

    def log_displacement_event(self, event):
        self.data_displacement_events.append(event)
        self.displaced_object_ids.add(event["obj_id"])
        if event["obj_id"] in self.tracked_objects:
            self.tracked_objects[event["obj_id"]]["displaced"] = True

    def log_displacement_debug(self, entry, verbose=False):
        """Record why a displacement attempt succeeded or failed."""
        row = dict(entry)
        row.setdefault("timestep", self.timestep)
        self.data_displacement_debug.append(row)
        # Avoid printing every step (freezes the terminal on long episodes)
        if verbose or row.get("status") == "ok":
            stage = row.get("stage", "?")
            status = row.get("status", "?")
            oid = row.get("obj_id", "")
            detail = row.get("detail", "")
            print(f"[displace] t={self.timestep} {status} stage={stage} obj={oid} {detail}")

    def get_dict_displacement_debug(self):
        cols = {
            "timestep": [],
            "obj_id": [],
            "status": [],
            "stage": [],
            "detail": [],
            "room_id": [],
            "from_receptacle": [],
            "to_receptacle": [],
            "n_receptacles_room": [],
            "n_receptacles_tried": [],
            "n_skipped_same_parent": [],
            "n_skipped_closed": [],
            "n_spawn_empty": [],
            "n_place_fail": [],
            "last_error": [],
        }
        for entry in self.data_displacement_debug:
            cols["timestep"].append(entry.get("timestep"))
            cols["obj_id"].append(entry.get("obj_id"))
            cols["status"].append(entry.get("status"))
            cols["stage"].append(entry.get("stage"))
            cols["detail"].append(entry.get("detail"))
            cols["room_id"].append(entry.get("room_id"))
            cols["from_receptacle"].append(entry.get("from_receptacle"))
            cols["to_receptacle"].append(entry.get("to_receptacle"))
            cols["n_receptacles_room"].append(entry.get("n_receptacles_room"))
            cols["n_receptacles_tried"].append(entry.get("n_receptacles_tried"))
            cols["n_skipped_same_parent"].append(entry.get("n_skipped_same_parent"))
            cols["n_skipped_closed"].append(entry.get("n_skipped_closed"))
            cols["n_spawn_empty"].append(entry.get("n_spawn_empty"))
            cols["n_place_fail"].append(entry.get("n_place_fail"))
            cols["last_error"].append(entry.get("last_error"))
        return cols

    def set_world_layout(self, layout):
        self.world_layout = layout

    def _csv_path(self, table: str) -> str:
        return os.path.join(self.annotations_dir, f"{table}-{self.scene_name}.csv")

    def _json_path(self, name: str) -> str:
        return os.path.join(self.annotations_dir, f"{name}-{self.scene_name}.json")

    def _append_table(self, table: str, columns: dict) -> None:
        """Append column-oriented rows to a CSV; write header only once."""
        if not columns or not next(iter(columns.values())):
            return
        os.makedirs(self.annotations_dir, exist_ok=True)
        path = self._csv_path(table)
        df = pd.DataFrame(columns)
        write_header = table not in self._csv_initialized
        df.to_csv(path, mode="a", header=write_header, index=False)
        self._csv_initialized.add(table)

    def flush_step_buffers(self) -> None:
        """Write growing step tables to disk and free RAM for those buffers."""
        if self.dict_agent:
            self._append_table("navigation", self.get_dict_navigation())
            self.dict_agent.clear()
        if self.data_doors:
            self._append_table("doors", self.get_dict_doors())
            self._append_table("passage_state", self.get_dict_passage_state())
            self.data_doors.clear()
        if self.data_object_state:
            self._append_table("object_state", self.get_dict_object_state())
            self.data_object_state.clear()
        if self.data_region_trajectory:
            self._append_table("region_trajectory", self.get_dict_region_trajectory())
            self.data_region_trajectory.clear()
        if self.data_displacement_debug:
            self._append_table("displacement_debug", self.get_dict_displacement_debug())
            self.data_displacement_debug.clear()

    def collect_data(
        self,
        event,
        action,
        v_objects,
        controller,
        room_info=None,
        door_states=None,
        action_success=None,
        held_obj_id=None,
        object_states=None,
    ):
        # Truncate like an LLM context window: keep only the first max_steps frames
        if self.at_capacity:
            if not self._truncated:
                self._truncated = True
                print(
                    f"[collector] max_steps={self.max_steps} reached; "
                    "further frames are ignored"
                )
            return

        if not self.dict_colors:
            self.dict_colors = {d["name"]: d["color"] for d in event.metadata["colors"]}

        position = self.round_number(event.metadata["agent"]["position"], 2)
        rotation = self.round_number(event.metadata["agent"]["rotation"], 2)
        camera_position = self.round_number(event.metadata["cameraPosition"], 2)
        camera_horizon = np.round(event.metadata["agent"]["cameraHorizon"], 2)
        action_name = THORActions.get_action_name(action)
        key = (self.timestep, action_name)

        room_info = room_info or {}
        current_room = room_info.get("current_room")
        current_room_type = room_info.get("current_room_type")
        seen_rooms = room_info.get("seen_rooms", [])
        room_just_entered = False
        if current_room is not None and current_room not in self.visited_rooms:
            self.visited_rooms.add(current_room)
            room_just_entered = True

        if key not in self.dict_agent:
            self.dict_agent[key] = {"objects": []}
            self.dict_agent[key]["position"] = position
            self.dict_agent[key]["rotation"] = rotation
            self.dict_agent[key]["degrees"] = AGENT_ROTATION_DEG
            self.dict_agent[key]["camera_horizon"] = camera_horizon
            self.dict_agent[key]["camera_position"] = camera_position
            self.dict_agent[key]["current_room"] = current_room
            self.dict_agent[key]["current_room_type"] = current_room_type
            self.dict_agent[key]["room_just_entered"] = room_just_entered
            self.dict_agent[key]["visited_rooms"] = ";".join(
                sorted(str(r) for r in self.visited_rooms)
            )
            self.dict_agent[key]["seen_rooms"] = ";".join(str(r) for r in seen_rooms)
            self.dict_agent[key]["action_success"] = (
                bool(action_success) if action_success is not None else None
            )
            self.dict_agent[key]["held_obj_id"] = held_obj_id
            cond_objs, objects = self.get_object_data(v_objects, controller)
            self.dict_agent[key]["objects"] = cond_objs
            if self.data_objects["objects"]:
                self.data_objects["objects"].update(objects)
            else:
                self.data_objects["objects"] = objects

            if door_states:
                for door in door_states:
                    self.data_doors.append(
                        {
                            "timestep": self.timestep,
                            "door_id": door["door_id"],
                            "room0": door["room0"],
                            "room1": door["room1"],
                            "openable": door["openable"],
                            "is_open": door["is_open"],
                            "openness": door["openness"],
                            "position": door["position"],
                            "distance": door["distance"],
                        }
                    )

            self.data_region_trajectory.append(
                {
                    "timestep": self.timestep,
                    "region_id": current_room,
                    "region_type": current_room_type,
                }
            )

            if object_states:
                for row in object_states:
                    self.log_object_state_row(
                        timestep=self.timestep,
                        obj_meta=row["obj_meta"],
                        in_camera_fov=row["in_camera_fov"],
                        receptacle_is_open=row.get("receptacle_is_open"),
                    )

            image_name = os.path.join(self.image_path, "img_" + str(self.timestep) + ".png")
            self.dict_agent[key]["image"] = image_name
            self.save_image(image_name, event)
            self.timestep += 1

            if self.timestep % self.flush_every == 0:
                self.flush_step_buffers()

    def save_data(self, reason="done"):
        """Finalize episode exports. Safe to call more than once (idempotent).

        Always call on agent ``done`` *or* when the episode hits ``max_steps``.
        """
        if self._saved:
            return
        self._save_reason = reason
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Drain any remaining step buffers (also creates empty-capable streams)
        self.flush_step_buffers()

        # Ensure stream tables exist even if empty (stable layout for consumers)
        for table in self._STREAM_TABLES:
            path = self._csv_path(table)
            if not os.path.exists(path):
                empty = {
                    "navigation": self.get_dict_navigation,
                    "doors": self.get_dict_doors,
                    "object_state": self.get_dict_object_state,
                    "region_trajectory": self.get_dict_region_trajectory,
                    "passage_state": self.get_dict_passage_state,
                    "displacement_debug": self.get_dict_displacement_debug,
                }[table]()
                pd.DataFrame(empty).to_csv(path, index=False)
                self._csv_initialized.add(table)

        pd.DataFrame(self.get_dict_objects()).to_csv(
            self._csv_path("objects"), index=False
        )
        pd.DataFrame(self.get_dict_displacement_events()).to_csv(
            self._csv_path("displacement_events"), index=False
        )

        if self.world_layout is not None:
            layout = dict(self.world_layout)
            layout["scene_id"] = self.scene_name
            layout["episode_id"] = self.episode_id
            with open(self._json_path("world_layout"), "w") as f:
                json.dump(layout, f, indent=2)

        fail_counts = {}
        debug_path = self._csv_path("displacement_debug")
        if os.path.exists(debug_path):
            try:
                debug_df = pd.read_csv(debug_path)
                if len(debug_df) and "stage" in debug_df.columns:
                    fail_counts = (
                        debug_df[debug_df["status"] == "fail"]["stage"]
                        .value_counts()
                        .to_dict()
                    )
            except Exception:
                fail_counts = {}

        meta = {
            "episode_id": self.episode_id,
            "episode_kind": self.episode_kind,
            "environment": self.environment,
            "scene_id": self.scene_name,
            "num_timesteps": self.timestep,
            "max_steps": self.max_steps,
            "truncated": self._truncated or self.at_capacity,
            "save_reason": reason,
            "num_displacements": len(self.data_displacement_events),
            "num_tracked_objects": len(self.tracked_objects),
            "displacement_fail_stages": fail_counts,
            "images_dir": "images",
            "annotations_dir": "annotations",
            "camera": {
                "width": self.FRAME_WIDTH,
                "height": self.FRAME_HEIGHT,
                "frame_size_px": self.FRAME_SIZE_PX,
                "fov_vertical_deg": self.VERTICAL_FOV_DEG,
                "source": "nav (INTEL) camera",
            },
            "agent": {
                "movement_constant": AGENT_MOVEMENT_CONSTANT,
                "rotation_deg": AGENT_ROTATION_DEG,
                "horizon_deg": HORIZON,
                "arm_move_constant": ARM_MOVE_CONSTANT,
                "wrist_rotation_deg": WRIST_ROTATION,
            },
            "visibility_filters": {
                "visibility_mode": self.visibility_mode,
                "export_policy": "named_non_structural_with_metrics",
                "suggested_postprocess_thresholds": {
                    "min_frame_fraction": self.min_frame_fraction,
                    "min_mask_pixels": self.min_mask_pixels,
                    "min_bbox_side": self.min_bbox_side,
                    "min_unoccluded_ratio": self.min_unoccluded_ratio,
                },
                "note": (
                    "navigation/objects CSV lists named non-structural FOV objects "
                    "(drops Wall/Floor and numeric-only ids like '2|4'); "
                    "filter with suggested thresholds in post-processing. "
                    "Displacement discovery still uses filter_recognizable_detections. "
                    "Use camera.width/height and camera.fov_vertical_deg when recomputing "
                    "visible-frac or full-silhouette-px offline."
                ),
            },
        }
        with open(self._json_path("episode_meta"), "w") as f:
            json.dump(meta, f, indent=2)

        self._saved = True
        print(
            f"[collector] saved episode ({reason}) timesteps={self.timestep} "
            f"dir={self.out_dir} (images/ + annotations/)"
        )
