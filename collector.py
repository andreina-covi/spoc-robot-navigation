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

from geometry import (
    DEFAULT_NAV_CAMERA_MOUNT_DEG,
    angular_size_deg,
    box_center_size,
    camera_intrinsics,
    corners_from_box,
    expected_bbox_from_3d,
)


class Collector:
    """Logs navigation + cm-benchmark fields for invisible_displacement / survey.

    Hard-caps logging at ``max_steps`` (like an LLM context window): once that many
    timesteps are recorded, further ``collect_data`` calls are no-ops. Step buffers
    are flushed to CSV periodically so long episodes do not keep the full table in RAM.

    Episode layout under ``OBJAVERSE_NAVIGATION_PATH/<timestamp>/``:
    - ``images/`` — RGB frames
    - ``annotations/`` — all CSV and JSON exports
    """

    FRAME_WIDTH = INTEL_CAMERA_WIDTH
    FRAME_HEIGHT = INTEL_CAMERA_HEIGHT
    FRAME_SIZE_PX = FRAME_WIDTH * FRAME_HEIGHT
    VERTICAL_FOV_DEG = INTEL_VERTICAL_FOV

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
    ):
        self.dict_agent = {}
        self.data_objects = {}  # objectId -> catalog tuple for objects-*.csv
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

        self.tracked_objects = {}
        self.displaced_object_ids = set()
        self.world_layout = None
        # How often RoomVisit enables renderImageSynthesis for FOV object logging
        self.fov_stride = 1
        # Synced from StretchController.calibrate_agent (RotateCameraMount / FOV)
        self.nav_camera_mount_deg = DEFAULT_NAV_CAMERA_MOUNT_DEG
        self.nav_camera_fov_deg = float(INTEL_VERTICAL_FOV)

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

    def get_visible_pixels_from_bbox(self, event, bbox, color):
        """Get visible pixels from bbox and color."""
        cmin, rmin, cmax, rmax = bbox
        crop = event.instance_segmentation_frame[rmin:rmax, cmin:cmax]
        visible_pixels = int(np.all(crop == color, axis=2).sum())
        return visible_pixels

    def get_object_data(self, arr_objects, event, include_detections=False):
        """Collect named non-structural objects for navigation / objects CSV.

        Always emit ``obj-id`` and ``obj-distance`` from metadata (no detections needed).
        Bbox / visible-pixels / occupancy are filled **only** when
        ``include_detections=True`` and ``event`` is the navigation step that actually
        ran ``renderImageSynthesis``.

        Important: use the nav ``event`` snapshot — do **not** read
        ``controller.last_event`` after later PlaceObjectAtPoint / Pass steps, which
        run without synthesis and wipe or stale-out detections.
        """
        objects = {}
        cond_objs = []
        detections = event.instance_detections2D if include_detections else None
        has_detections = detections is not None
        seg_frame = event.instance_segmentation_frame if include_detections else None

        # Camera pose for 3D→2D expected bbox.
        # If cameraHorizon already reflects the mount (non-zero after calibrate),
        # use it alone — do not add nav_camera_mount_deg again.
        meta = event.metadata
        cam_pos = meta.get("cameraPosition") or meta["agent"]["position"]
        yaw_deg = float(meta["agent"]["rotation"]["y"])
        horizon = float(meta["agent"].get("cameraHorizon", 0.0) or 0.0)
        if abs(horizon) > 1e-3:
            pitch_deg = horizon
        else:
            pitch_deg = float(self.nav_camera_mount_deg)
        fov_v = float(self.nav_camera_fov_deg or self.VERTICAL_FOV_DEG)
        f_px, _fov_h = camera_intrinsics(
            self.FRAME_WIDTH, self.FRAME_HEIGHT, fov_v
        )

        for obj_dict in arr_objects:
            oid = obj_dict["objectId"]
            obj_type = obj_dict.get("objectType")
            if not is_exportable_object(obj_type=obj_type, object_id=oid):
                continue

            color = self.dict_colors.get(oid)
            if color is None:
                continue

            # Prefer OOBB corners for projection; AABB for catalog center/size.
            oobb = obj_dict.get("objectOrientedBoundingBox")
            aabb = obj_dict.get("axisAlignedBoundingBox")
            box = None
            if corners_from_box(oobb):
                box = oobb
            elif corners_from_box(aabb):
                box = aabb
            else:
                box = oobb if oobb is not None else (aabb if aabb is not None else {})

            expected_bbox = expected_bbox_from_3d(
                box,
                cam_pos,
                yaw_deg,
                pitch_deg,
                self.FRAME_WIDTH,
                self.FRAME_HEIGHT,
                f_px,
                clamp_to_image=True,
            )
            if expected_bbox is not None:
                expected_bbox_area = float(expected_bbox["area"])
                expected_cmin = float(expected_bbox["cmin"])
                expected_cmax = float(expected_bbox["cmax"])
                expected_rmin = float(expected_bbox["rmin"])
                expected_rmax = float(expected_bbox["rmax"])
                ang_width_deg, ang_height_deg = angular_size_deg(
                    expected_cmin,
                    expected_cmax,
                    expected_rmin,
                    expected_rmax,
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    f_px,
                )
            else:
                expected_bbox_area = None
                expected_cmin = None
                expected_cmax = None
                expected_rmin = None
                expected_rmax = None
                ang_width_deg = None
                ang_height_deg = None

            # Catalog / objects-*.csv — 3D metadata (not segmentation).
            # AABB center/size; never default to zeros from an OOBB without those fields.
            center, size = box_center_size(aabb=aabb, oobb=oobb)
            objects[oid] = (
                obj_type,
                oid,
                tuple(color),
                tuple(self.round_number(obj_dict["position"], 2)),
                tuple(self.round_number(obj_dict["rotation"], 2)),
                tuple(obj_dict["receptacleObjectIds"])
                if obj_dict.get("receptacleObjectIds") is not None
                else (),
                tuple(self.round_number(center, 2)),
                tuple(self.round_number(size, 2)),
            )

            dist = float(obj_dict.get("distance") or 0.0)
            bbox = None
            visible_pixels = None
            bbox_area = None
            min_side = None
            occupancy = None

            # Pixel metrics only on synthesis-stride steps (from the nav event)
            if (
                include_detections
                and has_detections
                and seg_frame is not None
                and oid in detections
            ):
                cand = detections[oid]
                visible_pixels = self.get_visible_pixels_from_bbox(event, cand, color)
                if visible_pixels > 0:
                    cmin, rmin, cmax, rmax = cand
                    bbox_area = (cmax - cmin) * (rmax - rmin)
                    if bbox_area > 0:
                        bbox = cand
                        min_side = min(cmax - cmin, rmax - rmin)
                        occupancy = np.round(visible_pixels / bbox_area, 3)
                    else:
                        visible_pixels = None
                        bbox_area = None
                else:
                    visible_pixels = None

            # Always a nav row: id + distance every step; 3D-expected metrics when
            # projectable; detection metrics may be None off-stride / out of FOV.
            # Layout: [oid, dist, det_bbox, exp_area, ang_w, ang_h,
            #          exp_cmin, exp_cmax, exp_rmin, exp_rmax,
            #          vis_px, det_area, min_side, occ, displaced]
            was_displaced = oid in self.displaced_object_ids
            cond_objs.append(
                [
                    oid,
                    np.round(dist, 4),
                    bbox if bbox is not None else [None, None, None, None],
                    None
                    if expected_bbox_area is None
                    else float(np.round(expected_bbox_area, 2)),
                    None
                    if ang_width_deg is None
                    else float(np.round(ang_width_deg, 2)),
                    None
                    if ang_height_deg is None
                    else float(np.round(ang_height_deg, 2)),
                    None
                    if expected_cmin is None
                    else float(np.round(expected_cmin, 2)),
                    None
                    if expected_cmax is None
                    else float(np.round(expected_cmax, 2)),
                    None
                    if expected_rmin is None
                    else float(np.round(expected_rmin, 2)),
                    None
                    if expected_rmax is None
                    else float(np.round(expected_rmax, 2)),
                    visible_pixels,
                    bbox_area,
                    min_side,
                    occupancy,
                    was_displaced,
                ]
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
            dict_navigation["expected-bbox-area"].append(None)
            dict_navigation["ang-width-deg"].append(None)
            dict_navigation["ang-height-deg"].append(None)
            dict_navigation["expected_cmin"].append(None)
            dict_navigation["expected_cmax"].append(None)
            dict_navigation["expected_rmin"].append(None)
            dict_navigation["expected_rmax"].append(None)
            dict_navigation["visible-pixels"].append(None)
            dict_navigation["bbox-area"].append(None)
            dict_navigation["min-side"].append(None)
            dict_navigation["occupancy-ratio"].append(None)
            dict_navigation["displaced"].append(None)
            return

        for object_data in objects_data:
            self.add_basic_navigation_data(dict_navigation, key)
            dict_navigation["obj-id"].append(object_data[0])
            dict_navigation["obj-distance"].append(object_data[1])
            bbox = object_data[2] if len(object_data) > 2 else [None, None, None, None]
            if bbox is None:
                bbox = [None, None, None, None]
            self.save_bbox(dict_navigation, bbox)
            # Layout: [oid, dist, bbox, exp_area, ang_w, ang_h,
            #          exp_cmin, exp_cmax, exp_rmin, exp_rmax,
            #          vis_px, area, min_side, occ, displaced]
            if len(object_data) >= 15:
                dict_navigation["expected-bbox-area"].append(object_data[3])
                dict_navigation["ang-width-deg"].append(object_data[4])
                dict_navigation["ang-height-deg"].append(object_data[5])
                dict_navigation["expected_cmin"].append(object_data[6])
                dict_navigation["expected_cmax"].append(object_data[7])
                dict_navigation["expected_rmin"].append(object_data[8])
                dict_navigation["expected_rmax"].append(object_data[9])
                dict_navigation["visible-pixels"].append(object_data[10])
                dict_navigation["bbox-area"].append(object_data[11])
                dict_navigation["min-side"].append(object_data[12])
                dict_navigation["occupancy-ratio"].append(object_data[13])
                dict_navigation["displaced"].append(bool(object_data[14]))
            elif len(object_data) >= 14:
                dict_navigation["expected-bbox-area"].append(object_data[3])
                dict_navigation["ang-width-deg"].append(object_data[4])
                dict_navigation["ang-height-deg"].append(object_data[5])
                dict_navigation["expected_cmin"].append(object_data[6])
                dict_navigation["expected_cmax"].append(object_data[7])
                dict_navigation["expected_rmin"].append(object_data[8])
                dict_navigation["expected_rmax"].append(object_data[9])
                dict_navigation["visible-pixels"].append(object_data[10])
                dict_navigation["bbox-area"].append(object_data[11])
                dict_navigation["min-side"].append(object_data[12])
                dict_navigation["occupancy-ratio"].append(object_data[13])
                dict_navigation["displaced"].append(None)
            else:
                # Legacy shorter rows
                dict_navigation["expected-bbox-area"].append(
                    object_data[3] if len(object_data) > 3 else None
                )
                dict_navigation["ang-width-deg"].append(
                    object_data[4] if len(object_data) > 4 else None
                )
                dict_navigation["ang-height-deg"].append(
                    object_data[5] if len(object_data) > 5 else None
                )
                dict_navigation["expected_cmin"].append(None)
                dict_navigation["expected_cmax"].append(None)
                dict_navigation["expected_rmin"].append(None)
                dict_navigation["expected_rmax"].append(None)
                if len(object_data) >= 10:
                    dict_navigation["visible-pixels"].append(object_data[6])
                    dict_navigation["bbox-area"].append(object_data[7])
                    dict_navigation["min-side"].append(object_data[8])
                    dict_navigation["occupancy-ratio"].append(object_data[9])
                else:
                    dict_navigation["visible-pixels"].append(None)
                    dict_navigation["bbox-area"].append(None)
                    dict_navigation["min-side"].append(None)
                    dict_navigation["occupancy-ratio"].append(None)
                dict_navigation["displaced"].append(None)

    def save_image(self, im_path, event):
        cv2.imwrite(im_path, event.cv2img)

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
            "cmin": [],
            "rmin": [],
            "cmax": [],
            "rmax": [],
            "expected-bbox-area": [],
            "ang-width-deg": [],
            "ang-height-deg": [],
            "expected_cmin": [],
            "expected_cmax": [],
            "expected_rmin": [],
            "expected_rmax": [],
            "visible-pixels": [],
            "bbox-area": [],
            "min-side": [],
            "occupancy-ratio": [],
            "displaced": [],
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
        for t in self.data_objects.values():
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

    def update_visibility_tracking(self, visible_pickupable_meta):
        """Track pickupables using THOR ``visible`` (not instance_detections2D).

        ``visible_pickupable_meta`` maps objectId → metadata for pickupables with
        ``visible=True`` this step (within ``visibilityDistance``).
        """
        visible_ids = set(visible_pickupable_meta.keys())
        for oid, track in self.tracked_objects.items():
            if oid in visible_ids:
                track["last_visible_t"] = self.timestep
                track["hidden_steps"] = 0
            else:
                track["hidden_steps"] += 1
        for oid, meta in visible_pickupable_meta.items():
            if oid in self.tracked_objects:
                continue
            self.tracked_objects[oid] = {
                "obj_type": meta.get("objectType"),
                "first_seen_t": self.timestep,
                "last_visible_t": self.timestep,
                "hidden_steps": 0,
                "displaced": False,
            }

    def candidates_for_displacement(self):
        """Pickupables seen before (visible once), now not visible for ≥2 steps, not moved."""
        if len(self.displaced_object_ids) >= self.max_displacements:
            return []
        candidates = []
        for oid, track in self.tracked_objects.items():
            if track["displaced"] or oid in self.displaced_object_ids:
                continue
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
        include_detections=False,
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
            # Nav: id+distance every step; bbox/mask only when synthesis ran.
            # Use the nav ``event`` (not last_event) so displace PlaceObject steps
            # cannot wipe stride detections.
            cond_objs, objects = self.get_object_data(
                v_objects, event, include_detections=include_detections
            )
            self.dict_agent[key]["objects"] = cond_objs
            # Prefer non-zero center/size when merging catalog rows for the same oid
            for oid, row in objects.items():
                prev = self.data_objects.get(oid)
                if prev is None:
                    self.data_objects[oid] = row
                    continue
                prev_size = sum(abs(float(v)) for v in prev[7])
                new_size = sum(abs(float(v)) for v in row[7])
                if new_size >= prev_size:
                    self.data_objects[oid] = row

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

            image_name = "img_" + str(self.timestep) + ".png"
            im_path = os.path.join(self.image_path, image_name)
            self.dict_agent[key]["image"] = image_name
            self.save_image(im_path, event)
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
            "path_to_data": self.out_dir,
            "path_to_images": self.image_path,
            "path_to_annotations": self.annotations_dir,
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
            "fov_stride": self.fov_stride,
            "camera": {
                "width": self.FRAME_WIDTH,
                "height": self.FRAME_HEIGHT,
                "frame_size_px": self.FRAME_SIZE_PX,
                "fov_vertical_deg": self.nav_camera_fov_deg,
                "nav_camera_mount_deg": self.nav_camera_mount_deg,
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
                "note": (
                    "navigation/objects CSV lists named non-structural FOV objects "
                    "(drops Wall/Floor and numeric-only ids like '2|4')."
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
