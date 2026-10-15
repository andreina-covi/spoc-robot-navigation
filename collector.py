import os
import json
import datetime
import subprocess

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
    robot_action_by_name,
)
from utils.constants.object_constants import is_exportable_object
from utils.object_labels import describe_object
from utils.type_utils import THORActions

from geometry import (
    DEFAULT_NAV_CAMERA_MOUNT_DEG,
    aabb_bounds,
    angular_size_deg,
    box_center_size,
    camera_intrinsics,
    corners_from_box,
    expected_bbox_from_3d,
)


class Collector:
    """Logs navigation, placement trials, and survey fields.

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
        "passage_state",
    )

    def __init__(
        self,
        scene_name=None,
        episode_kind="invisible_displacement",
        environment="ai2thor",
        max_steps=None,
        flush_every=50,
    ):
        self.dict_agent = {}
        self.data_objects = {}  # objectId -> catalog dict for objects-*.csv
        self.house_index = None
        self.house_split = None
        self.move_distances_m = set()
        self.rotation_angles_deg = set()
        self.initial_door_states = []
        self.data_doors = []
        self.data_object_state = []
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
        # Context-window style cap: keep at most this many logged steps
        self.max_steps = max_steps
        self.flush_every = max(1, int(flush_every))
        self._saved = False
        self._csv_initialized = set()
        self._truncated = False
        self._save_reason = None

        self.tracked_objects = {}
        self.world_layout = None
        # AI2-THOR reachability graphs (independent of SPOC trajectory)
        self.nav_graphs = {}  # snapshot -> graph dict
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
        """Detections for CSV export: non-structural objects, including numeric ids.

        Visibility thresholds are **not** applied — metrics go to CSV for post-processing.
        Drops Wall/Floor/Ceiling/Room. Ids such as ``6|5`` are kept.
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
        """Collect non-structural objects for navigation / objects CSV.

        Navigation rows are emitted **only** when the object has mask pixels in the
        current nav frame (``visible-pixels > 0``). Hidden / tracked-only objects belong in ``object_state``, not
        navigation.

        Bbox metrics require ``include_detections=True`` and the nav ``event`` that
        ran ``renderImageSynthesis`` (do not use ``controller.last_event`` after
        later PlaceObjectAtPoint / Pass steps).
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

            # Require in-image mask pixels for a navigation row (clean FOV export)
            visible_pixels = None
            bbox = None
            bbox_area = None
            min_side = None
            occupancy = None
            if not (
                include_detections
                and has_detections
                and seg_frame is not None
                and oid in detections
            ):
                continue
            cand = detections[oid]
            visible_pixels = self.get_visible_pixels_from_bbox(event, cand, color)
            if visible_pixels is None or visible_pixels <= 0:
                continue
            cmin, rmin, cmax, rmax = cand
            bbox_area = (cmax - cmin) * (rmax - rmin)
            if bbox_area <= 0:
                continue
            bbox = cand
            min_side = min(cmax - cmin, rmax - rmin)
            occupancy = np.round(visible_pixels / bbox_area, 3)

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
            label = describe_object(oid, obj_type, obj_dict.get("assetId"))
            center, size = box_center_size(aabb=aabb, oobb=oobb)
            aabb_min, aabb_max = aabb_bounds(aabb)
            objects[oid] = {
                "obj_type": label.obj_type,
                "obj_id": oid,
                "obj_synset": label.synset,
                "asset_id": obj_dict.get("assetId") or "",
                "color": tuple(color),
                "position": tuple(self.round_number(obj_dict["position"], 2)),
                "rotation": tuple(self.round_number(obj_dict["rotation"], 2)),
                "receptacles": (
                    tuple(obj_dict["receptacleObjectIds"])
                    if obj_dict.get("receptacleObjectIds") is not None
                    else ()
                ),
                "center": tuple(self.round_number(center, 2)),
                "size": tuple(self.round_number(size, 2)),
                "aabb_min": self._xyz(aabb_min),
                "aabb_max": self._xyz(aabb_max),
                "room_id": obj_dict.get("roomId"),
            }

            dist = float(obj_dict.get("distance") or 0.0)
            cond_objs.append(
                [
                    oid,
                    np.round(dist, 4),
                    bbox,
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
                ]
            )

        return cond_objs, objects

    def _xyz(self, point):
        if not point:
            return (None, None, None)
        return tuple(self.round_number(point, 2))

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
            #          vis_px, area, min_side, occ]
            if len(object_data) >= 14:
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
            "path": [],
        }
        for key in self.dict_agent:
            self.save_data_navigation(dict_navigation, key)
        return dict_navigation

    def get_dict_objects(self):
        dict_objects = {
            "obj-type": [],
            "obj-synset": [],
            "obj-id": [],
            "assetId": [],
            "room-id": [],
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
            "aabb-min-x": [],
            "aabb-min-y": [],
            "aabb-min-z": [],
            "aabb-max-x": [],
            "aabb-max-y": [],
            "aabb-max-z": [],
        }
        for row in self.data_objects.values():
            dict_objects["obj-type"].append(row["obj_type"])
            dict_objects["obj-synset"].append(row["obj_synset"])
            dict_objects["obj-id"].append(row["obj_id"])
            dict_objects["assetId"].append(row["asset_id"])
            dict_objects["room-id"].append(row["room_id"])
            dict_objects["obj-color"].append(row["color"])
            self.save_data_by_axis(dict_objects, "obj-pos", row["position"])
            self.save_data_by_axis(dict_objects, "obj-rot", row["rotation"])
            dict_objects["receptacleObjectIds"].append(row["receptacles"])
            self.save_data_by_axis(dict_objects, "bBox-center", row["center"])
            self.save_data_by_axis(dict_objects, "size", row["size"])
            self.save_data_by_axis(dict_objects, "aabb-min", row["aabb_min"])
            self.save_data_by_axis(dict_objects, "aabb-max", row["aabb_max"])
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
            "obj-synset": [],
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
            cols["obj-synset"].append(entry.get("obj_synset"))
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

    def get_dict_passage_state(self):
        """Survey-oriented view of door open/closed state."""
        cols = {
            "timestep": [],
            "passage_id": [],
            "obj-id": [],
            "is_open": [],
            "from_region": [],
            "to_region": [],
        }
        for entry in self.data_doors:
            cols["timestep"].append(entry["timestep"])
            cols["passage_id"].append(entry["door_id"])
            cols["obj-id"].append(entry["door_id"])
            cols["is_open"].append(entry["is_open"])
            cols["from_region"].append(entry["room0"])
            cols["to_region"].append(entry["room1"])
        return cols

    def record_initial_doors(self, door_states):
        """Store each door's open/closed state once. RoomVisit does not change doors."""
        if self.initial_door_states:
            return
        self.initial_door_states = [dict(door) for door in door_states or []]
        for door in self.initial_door_states:
            self.data_doors.append(
                {
                    "timestep": 0,
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

    def update_visibility_tracking(self, in_image_pickupable_meta):
        """Track pickupables by **nav-image** presence (detections + mask pixels).

        ``in_image_pickupable_meta`` maps objectId → metadata for pickupables that
        appear in the current agent RGB/seg frame (same signal as navigation
        ``visible-pixels > 0``). Call only on synthesis strides; off-stride callers
        should skip this method so ``hidden_steps`` is not advanced blindly.
        """
        visible_ids = set(in_image_pickupable_meta.keys())
        for oid, track in self.tracked_objects.items():
            if oid in visible_ids:
                track["last_visible_t"] = self.timestep
                track["hidden_steps"] = 0
                track["in_camera_fov"] = True
            else:
                track["hidden_steps"] += 1
                track["in_camera_fov"] = False
        for oid, meta in in_image_pickupable_meta.items():
            if oid in self.tracked_objects:
                continue
            self.tracked_objects[oid] = {
                "obj_type": meta.get("objectType"),
                "first_seen_t": self.timestep,
                "last_visible_t": self.timestep,
                "hidden_steps": 0,
                "in_camera_fov": True,
            }

    def log_object_state_row(self, timestep, obj_meta, in_camera_fov, receptacle_is_open=None):
        parents = obj_meta.get("parentReceptacles") or []
        parent = parents[0] if parents else None
        pos = self.round_number(obj_meta["position"], 2)
        rot = self.round_number(obj_meta["rotation"], 2)
        dist = obj_meta.get("distance")
        label = describe_object(
            obj_meta["objectId"], obj_meta.get("objectType"), obj_meta.get("assetId")
        )
        self.data_object_state.append(
            {
                "timestep": timestep,
                "obj_id": obj_meta["objectId"],
                "obj_type": label.obj_type,
                "obj_synset": label.synset,
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

    def set_world_layout(self, layout):
        self.world_layout = layout

    def set_nav_graph(self, graph, snapshot=None):
        """Store a GetReachablePositions nav graph for one scene-state snapshot.

        ``snapshot`` is typically ``episode_start`` or ``episode_end``. Graphs are
        written in ``save_data`` as ``nav_graph-*.json`` (not the agent trajectory).
        """
        if graph is None:
            return
        g = dict(graph)
        snap = snapshot or g.get("snapshot") or "episode_start"
        g["snapshot"] = snap
        g["episode_id"] = self.episode_id
        g["scene_id"] = self.scene_name
        self.nav_graphs[snap] = g

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

    def collect_data(
        self,
        event,
        action,
        v_objects,
        controller,
        room_info=None,
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
            self._note_motion(action_name)
            self.dict_agent[key]["degrees"] = self._rotation_degrees(action_name)
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
                prev_size = sum(abs(float(v or 0)) for v in prev["size"])
                new_size = sum(abs(float(v or 0)) for v in row["size"])
                if new_size >= prev_size:
                    self.data_objects[oid] = row

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

    def _locomotion_scalar(self, action_name: str, thor_action: str):
        """Absolute move (m) or turn (deg) for this logged action, or None."""
        spec = robot_action_by_name.get(action_name) or {}
        scalar = None
        if spec.get("action") == thor_action:
            scalar = (spec.get("args") or {}).get("move_scalar")
        if scalar is None:
            return None
        return abs(float(scalar))

    def _note_motion(self, action_name: str) -> None:
        move = self._locomotion_scalar(action_name, "MoveAgent")
        if move is not None:
            self.move_distances_m.add(move)
        turn = self._locomotion_scalar(action_name, "RotateAgent")
        if turn is not None:
            self.rotation_angles_deg.add(turn)

    def _rotation_degrees(self, action_name: str):
        return self._locomotion_scalar(action_name, "RotateAgent")

    def _git_commit(self) -> str:
        root = os.path.dirname(os.path.abspath(__file__))
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""
        if dirty:
            return f"{commit}-dirty"
        return commit

    def _motion_check(self) -> dict:
        params = (self.nav_graphs.get("episode_start") or {}).get("params") or {}
        grid = params.get("grid_size")
        move = params.get("agent_move_m")
        rotation = params.get("agent_rotation_deg")
        moves = sorted(self.move_distances_m)
        angles = sorted(self.rotation_angles_deg)
        notes = []
        if moves and move is not None and any(abs(value - float(move)) > 1e-6 for value in moves):
            notes.append(f"move distances used {moves}; nav_graph agent_move_m is {move}")
        if angles and rotation is not None and any(abs(value - float(rotation)) > 1e-6 for value in angles):
            notes.append(
                f"rotation angles used {angles}; nav_graph agent_rotation_deg is {rotation}"
            )
        if grid is not None and move is not None and abs(float(grid) - float(move) * 0.75) > 1e-4:
            notes.append(f"grid_size {grid} is not 0.75 * agent_move_m {move}")
        return {
            "move_m_used": moves,
            "rotation_deg_used": angles,
            "nav_graph_grid_size": grid,
            "nav_graph_agent_move_m": move,
            "nav_graph_agent_rotation_deg": rotation,
            "matches": not notes,
            "report": "; ".join(notes),
        }

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
                    "passage_state": self.get_dict_passage_state,
                }[table]()
                pd.DataFrame(empty).to_csv(path, index=False)
                self._csv_initialized.add(table)

        pd.DataFrame(self.get_dict_objects()).to_csv(
            self._csv_path("objects"), index=False
        )

        if self.world_layout is not None:
            layout = dict(self.world_layout)
            layout["scene_id"] = self.scene_name
            layout["episode_id"] = self.episode_id
            with open(self._json_path("world_layout"), "w") as f:
                json.dump(layout, f, indent=2)

        if self.nav_graphs:
            nav_export = {
                "episode_id": self.episode_id,
                "scene_id": self.scene_name,
                "snapshots": self.nav_graphs,
                "notes": (
                    "AI2-THOR navigability (GetReachablePositions), independent of the "
                    "SPOC trajectory in navigation-*.csv. Prefer episode_start for the "
                    "environment at rollout begin; episode_end reflects doors/objects "
                    "after the episode. Do not treat the SPOC path as optimal."
                ),
            }
            with open(self._json_path("nav_graph"), "w") as f:
                json.dump(nav_export, f, indent=2)

        meta = {
            "path_to_data": self.out_dir,
            "path_to_images": self.image_path,
            "path_to_annotations": self.annotations_dir,
            "episode_id": self.episode_id,
            "episode_kind": self.episode_kind,
            "environment": self.environment,
            "scene_id": self.scene_name,
            "house_index": self.house_index,
            "split": self.house_split,
            "git_commit": self._git_commit(),
            "config": {
                "movement_m": AGENT_MOVEMENT_CONSTANT,
                "rotation_deg": AGENT_ROTATION_DEG,
                "small_rotation_deg": AGENT_ROTATION_DEG / 5,
                "reachability_grid_size": AGENT_MOVEMENT_CONSTANT * 0.75,
                "horizon_deg": HORIZON,
                "fov_stride": self.fov_stride,
                "max_steps": self.max_steps,
            },
            "num_timesteps": self.timestep,
            "max_steps": self.max_steps,
            "truncated": self._truncated or self.at_capacity,
            "save_reason": reason,
            "num_tracked_objects": len(self.tracked_objects),
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
                "reachability_grid_size": AGENT_MOVEMENT_CONSTANT * 0.75,
            },
            "nav_graph": {
                "file": f"nav_graph-{self.scene_name}.json",
                "snapshots": sorted(self.nav_graphs.keys()),
                "num_nodes_start": (
                    self.nav_graphs.get("episode_start", {}).get("num_nodes")
                ),
                "num_edges_start": (
                    self.nav_graphs.get("episode_start", {}).get("num_edges")
                ),
                "num_nodes_end": (
                    self.nav_graphs.get("episode_end", {}).get("num_nodes")
                ),
                "num_edges_end": (
                    self.nav_graphs.get("episode_end", {}).get("num_edges")
                ),
            },
            "visibility_filters": {
                "note": (
                    "navigation/objects CSV: non-structural objects with "
                    "nav mask pixels > 0. Numeric ids such as 6|5 are kept. "
                    "Wall/Floor/Ceiling/Room are dropped. "
                    "object_state.in_camera_fov uses the same mask-pixel signal "
                    "on synthesis strides; object_state.visible is THOR metadata."
                ),
            },
            "motion": self._motion_check(),
        }
        with open(self._json_path("episode_meta"), "w") as f:
            json.dump(meta, f, indent=2)

        self._saved = True
        motion = meta["motion"]
        print(
            f"[collector] saved episode ({reason}) timesteps={self.timestep} "
            f"dir={self.out_dir} (images/ + annotations/)"
        )
        if motion["report"]:
            print(f"[collector] nav_graph params differ from motion used: {motion['report']}")
