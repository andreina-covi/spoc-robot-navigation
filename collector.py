import os
import json
import datetime

import numpy as np
import pandas as pd
import cv2

from utils.constants.objaverse_data_dirs import OBJAVERSE_NAVIGATION_PATH
from utils.constants.stretch_initialization_utils import AGENT_ROTATION_DEG
from utils.type_utils import THORActions


class Collector:
    """Logs navigation + cm-benchmark fields for invisible_displacement / survey."""

    def __init__(
        self,
        scene_name=None,
        episode_kind="invisible_displacement",
        environment="ai2thor",
        max_displacements=5,
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
        self.image_path = os.path.join(OBJAVERSE_NAVIGATION_PATH, self.dt, "images")
        self.dict_colors = {}
        self.visited_rooms = set()
        self.max_displacements = max_displacements

        # Track objects seen in nav FOV for invisible displacement
        # oid -> {obj_type, first_seen_t, last_in_fov_t, hidden_steps, displaced}
        self.tracked_objects = {}
        self.displaced_object_ids = set()
        self.world_layout = None

        os.makedirs(self.image_path, exist_ok=True)

    def round_number(self, arr_numbers, n_round):
        if type(arr_numbers) is dict:
            rounded_arr = [arr_numbers["x"], arr_numbers["y"], arr_numbers["z"]]
        else:
            rounded_arr = arr_numbers
        return tuple([np.round(number, n_round) for number in rounded_arr])

    def get_min_by_axis(self, bbox):
        array = np.array(bbox)
        assert array.shape == (8, 3)
        x_min = np.min(array[:, 0])
        y_min = np.min(array[:, 1])
        z_min = np.min(array[:, 2])
        return (x_min, y_min, z_min)

    def get_object_data(self, arr_objects, controller, min_distance=0.0, max_distance=None):
        """Collect objects visible in the nav camera via instance_detections2D."""
        objects = set()
        cond_objs = []
        detections = controller.last_event.instance_detections2D
        if detections is None:
            return cond_objs, objects

        pos_dict_default = {"x": 0.0, "y": 0.0, "z": 0.0}
        for obj_dict in arr_objects:
            oid = obj_dict["objectId"]
            if oid not in detections:
                continue

            dist = float(obj_dict["distance"])
            if dist < min_distance:
                continue
            if max_distance is not None and dist > max_distance:
                continue

            bbox = detections[oid]
            cond_objs.append([oid, np.round(dist, 4), bbox])

            bbox_name = (
                "axisAlignedBoundingBox"
                if obj_dict["axisAlignedBoundingBox"] is not None
                else "objectOrientedBoundingBox"
            )
            color = self.dict_colors.get(oid)
            if color is None:
                continue
            objects.add(
                (
                    obj_dict["objectType"],
                    oid,
                    tuple(color),
                    tuple(self.round_number(obj_dict["position"], 2)),
                    tuple(self.round_number(obj_dict["rotation"], 2)),
                    tuple(obj_dict["receptacleObjectIds"])
                    if obj_dict["receptacleObjectIds"] is not None
                    else (),
                    tuple(self.round_number(obj_dict[bbox_name].get("center", pos_dict_default), 2)),
                    tuple(self.round_number(obj_dict[bbox_name].get("size", pos_dict_default), 2)),
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
        else:
            for object_data in objects_data:
                self.add_basic_navigation_data(dict_navigation, key)
                dict_navigation["obj-id"].append(object_data[0])
                dict_navigation["obj-distance"].append(object_data[1])
                self.save_bbox(dict_navigation, object_data[2])

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

    def save_data(self):
        out_dir = os.path.join(OBJAVERSE_NAVIGATION_PATH, self.dt)
        os.makedirs(out_dir, exist_ok=True)
        suffix = self.scene_name

        pd.DataFrame(self.get_dict_navigation()).to_csv(
            os.path.join(out_dir, f"navigation-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_objects()).to_csv(
            os.path.join(out_dir, f"objects-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_doors()).to_csv(
            os.path.join(out_dir, f"doors-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_object_state()).to_csv(
            os.path.join(out_dir, f"object_state-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_displacement_events()).to_csv(
            os.path.join(out_dir, f"displacement_events-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_displacement_debug()).to_csv(
            os.path.join(out_dir, f"displacement_debug-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_passage_state()).to_csv(
            os.path.join(out_dir, f"passage_state-{suffix}.csv")
        )
        pd.DataFrame(self.get_dict_region_trajectory()).to_csv(
            os.path.join(out_dir, f"region_trajectory-{suffix}.csv")
        )

        if self.world_layout is not None:
            layout = dict(self.world_layout)
            layout["scene_id"] = self.scene_name
            layout["episode_id"] = self.episode_id
            with open(os.path.join(out_dir, f"world_layout-{suffix}.json"), "w") as f:
                json.dump(layout, f, indent=2)

        debug_df = pd.DataFrame(self.get_dict_displacement_debug())
        fail_counts = {}
        if len(debug_df) and "stage" in debug_df.columns:
            fail_counts = (
                debug_df[debug_df["status"] == "fail"]["stage"].value_counts().to_dict()
            )
        meta = {
            "episode_id": self.episode_id,
            "episode_kind": self.episode_kind,
            "environment": self.environment,
            "scene_id": self.scene_name,
            "num_displacements": len(self.data_displacement_events),
            "num_tracked_objects": len(self.tracked_objects),
            "num_displacement_debug_rows": len(self.data_displacement_debug),
            "displacement_fail_stages": fail_counts,
        }
        with open(os.path.join(out_dir, f"episode_meta-{suffix}.json"), "w") as f:
            json.dump(meta, f, indent=2)
