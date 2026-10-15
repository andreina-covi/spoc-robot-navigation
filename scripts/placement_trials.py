#!/usr/bin/env python3
"""Trial-place every pickupable in a house, without agent navigation.

Loads one house into AI2-THOR. For each pickupable, places it once on:

- the closest floor spawn within the landmark radius, one point per landmark
  in that room
- the 5 most salient non-openable receptacles in that room

Each anchor uses one spawn coordinate, cached for the house. The object is
restored only when the place succeeds. Writes
``placement_candidates-house_XXXXXX.csv``.

The RoomVisit command already has these house indexes on its eval samples.
After the agent finishes, it passes that list to ``run``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai2thor.platform  # noqa: E402

from environment.stretch_controller import StretchController  # noqa: E402
from utils.constants.objaverse_data_dirs import (  # noqa: E402
    OBJAVERSE_HOUSES_DIR,
    OBJAVERSE_NAVIGATION_PATH,
)
from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS  # noqa: E402
from utils.placement_anchors import (  # noqa: E402
    FLOOR_ANCHOR_RADIUS_M,
    TOP_RECEPTACLES,
    floor_anchor_points,
    is_landmark_object,
    room_id_at,
    salient_receptacles,
    xyz_dict,
)

CSV_FIELDS = (
    "obj-id",
    "anchor-id",
    "anchor-kind",
    "pos-x",
    "pos-y",
    "pos-z",
    "success",
)


def house_name(house_index: int) -> str:
    return f"house_{int(house_index):06d}"


def houses_from_samples(samples) -> Dict[str, List[int]]:
    """Unique house indexes from the eval samples, grouped by split."""
    grouped: Dict[str, List[int]] = {}
    seen = set()
    for sample in samples:
        raw_index = sample.get("house_id", sample.get("house_index"))
        if raw_index in (None, "", "unknown"):
            continue
        index = int(raw_index)
        split = sample.get("house_split") or "val"
        key = (split, index)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(split, []).append(index)
    return grouped


def load_house(split: str, house_index: int, houses_dir: Path) -> dict:
    path = houses_dir / f"{split}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt") as handle:
        for line_index, line in enumerate(handle):
            if line_index == house_index:
                return json.loads(line)
    raise IndexError(f"{path} has no house at line index {house_index}")


def landmarks_from_house(house: dict, room_poly_map) -> List[dict]:
    landmarks = []
    for obj in house.get("objects") or []:
        obj_type = obj.get("objectType") or (obj.get("assetId") or "").split("_")[0]
        if not is_landmark_object(obj_type, obj.get("assetId") or obj.get("id") or ""):
            continue
        if not obj.get("id") or not obj.get("position"):
            continue
        landmarks.append(
            {
                "landmark_id": obj["id"],
                "obj-type": obj_type,
                "position": obj["position"],
                "region_id": room_id_at(room_poly_map, obj["position"]),
            }
        )
    return landmarks


class SpawnCache:
    """One ``GetSpawnCoordinatesAboveReceptacle`` call per receptacle id."""

    def __init__(self, controller: StretchController):
        self.controller = controller
        self._coords: Dict[str, list] = {}

    def coordinates(self, receptacle_id: str) -> list:
        if receptacle_id not in self._coords:
            try:
                self._coords[receptacle_id] = (
                    self.controller.get_locations_on_receptacle(receptacle_id) or []
                )
            except Exception:
                self._coords[receptacle_id] = []
        return self._coords[receptacle_id]

    def first(self, receptacle_id: str):
        coords = self.coordinates(receptacle_id)
        if not coords:
            return None
        return xyz_dict(coords[0])


def place_at(controller: StretchController, object_id: str, position, rotation=None) -> bool:
    kwargs = dict(
        action="PlaceObjectAtPoint",
        objectId=object_id,
        position=xyz_dict(position),
        forceKinematic=True,
        renderImageSynthesis=False,
    )
    if rotation is not None:
        kwargs["rotation"] = xyz_dict(rotation)
    event = controller.step(**kwargs)
    return bool(event.metadata.get("lastActionSuccess", False))


def resolved_position(controller: StretchController, object_id: str, fallback):
    try:
        obj = controller.get_object(object_id)
        pos = obj["position"]
        return (
            round(float(pos["x"]), 2),
            round(float(pos["y"]), 2),
            round(float(pos["z"]), 2),
        )
    except Exception:
        point = xyz_dict(fallback)
        return (round(point["x"], 2), round(point["y"], 2), round(point["z"], 2))


def trial_anchor(controller, object_id, anchor_id, anchor_kind, target, origin, origin_rot):
    """One place. Restore only when that place succeeds."""
    if target is None:
        return row(object_id, anchor_id, anchor_kind, None, False)
    ok = place_at(controller, object_id, target)
    if not ok:
        return row(object_id, anchor_id, anchor_kind, None, False)
    pose = resolved_position(controller, object_id, target)
    restored = place_at(controller, object_id, origin, origin_rot)
    if not restored:
        print(f"[placement] restore failed obj={object_id} anchor={anchor_id}")
    return row(object_id, anchor_id, anchor_kind, pose, True)


def row(object_id, anchor_id, anchor_kind, pose, success) -> dict:
    return {
        "obj-id": object_id,
        "anchor-id": anchor_id,
        "anchor-kind": anchor_kind,
        "pos-x": None if pose is None else pose[0],
        "pos-y": None if pose is None else pose[1],
        "pos-z": None if pose is None else pose[2],
        "success": bool(success),
    }


def scene_objects(controller: StretchController) -> List[dict]:
    with controller.include_object_metadata_context():
        objects = [dict(obj) for obj in controller.controller.last_event.metadata["objects"]]
    for obj in objects:
        obj["room_id"] = room_id_at(controller.room_poly_map, obj.get("position"))
    return objects


def floor_ids_in_room(objects, room_id) -> List[str]:
    ids = []
    for obj in objects:
        oid = obj.get("objectId")
        is_floor = obj.get("objectType") == "Floor" or (
            oid is not None and str(oid).startswith("Floor|")
        )
        if is_floor and obj.get("room_id") == room_id and oid:
            ids.append(oid)
    return ids


def floor_spawns(cache: SpawnCache, objects, room_id) -> list:
    spawns = []
    for floor_id in floor_ids_in_room(objects, room_id):
        spawns.extend(cache.coordinates(floor_id))
    return spawns


def trials_for_house(controller, house, radius_m, top_n) -> List[dict]:
    objects = scene_objects(controller)
    landmarks = landmarks_from_house(house, controller.room_poly_map)
    cache = SpawnCache(controller)
    floor_by_room: Dict[Any, list] = {}
    rows = []
    pickupables = [obj for obj in objects if obj.get("pickupable") and obj.get("objectId")]
    for index, obj in enumerate(pickupables, start=1):
        oid = obj["objectId"]
        room_id = obj.get("room_id")
        print(f"[placement] {index}/{len(pickupables)} {oid} room={room_id}")
        if not room_id:
            continue
        origin = obj.get("position")
        origin_rot = obj.get("rotation")
        if room_id not in floor_by_room:
            floor_by_room[room_id] = floor_spawns(cache, objects, room_id)
        room_landmarks = [
            landmark for landmark in landmarks if landmark.get("region_id") == room_id
        ]
        for anchor_id, point in floor_anchor_points(room_landmarks, floor_by_room[room_id], radius_m):
            rows.append(
                trial_anchor(controller, oid, anchor_id, "floor", point, origin, origin_rot)
            )
        for rec in salient_receptacles(objects, room_id, limit=top_n):
            rid = rec.get("objectId") or rec.get("id")
            rows.append(
                trial_anchor(
                    controller,
                    oid,
                    rid,
                    "receptacle",
                    cache.first(rid),
                    origin,
                    origin_rot,
                )
            )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def open_controller() -> StretchController:
    return StretchController(
        **STRETCH_ENV_ARGS,
        platform=ai2thor.platform.CloudRendering,
    )


def run(split: str, house_indices: List[int], output_dir: Path, radius_m: float, top_n: int) -> None:
    if OBJAVERSE_HOUSES_DIR is None:
        raise FileNotFoundError("OBJAVERSE_HOUSES_DIR is not set")
    houses_dir = Path(OBJAVERSE_HOUSES_DIR)
    houses = [(index, load_house(split, index, houses_dir)) for index in house_indices]
    controller = open_controller()
    try:
        for index, house in houses:
            name = house_name(index)
            print(f"[placement] loading {name}")
            controller.reset(house)
            rows = trials_for_house(controller, house, radius_m, top_n)
            path = output_dir / f"placement_candidates-{name}.csv"
            write_csv(path, rows)
            ok = sum(1 for item in rows if item["success"])
            print(f"[placement] wrote {path} rows={len(rows)} success={ok}")
    finally:
        controller.stop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--house_index", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for placement_candidates-house_XXXXXX.csv. "
        "Default: $OBJAVERSE_NAVIGATION_PATH/placement_trials",
    )
    parser.add_argument("--radius_m", type=float, default=FLOOR_ANCHOR_RADIUS_M)
    parser.add_argument("--top_receptacles", type=int, default=TOP_RECEPTACLES)
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else Path(OBJAVERSE_NAVIGATION_PATH) / "placement_trials"
    run(args.split, args.house_index, output, args.radius_m, args.top_receptacles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
