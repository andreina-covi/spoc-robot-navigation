"""Receptacle and floor anchors for per-house placement trials."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.constants.object_constants import is_structural_object
from utils.data_generation_utils.navigation_utils import get_room_id_from_location

FLOOR_ANCHOR_RADIUS_M = 1.2
TOP_RECEPTACLES = 5

LANDMARK_TYPES = {
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


def room_id_at(room_poly_map, position):
    """Room that contains this xz position. Missing positions stay unset."""
    if not position:
        return None
    return get_room_id_from_location(room_poly_map, position, verbose=False)


def receptacle_salience(rec: Dict[str, Any]) -> float:
    """Visual prominence from axis-aligned box volume. Larger is more salient."""
    box = rec.get("axisAlignedBoundingBox") or {}
    size = box.get("size")
    if not size:
        return 0.0
    if isinstance(size, dict):
        dims = (size.get("x", 0), size.get("y", 0), size.get("z", 0))
    else:
        dims = (size[0], size[1], size[2])
    return abs(float(dims[0])) * abs(float(dims[1])) * abs(float(dims[2]))


def is_landmark_object(obj_type: Optional[str], asset_or_id: str = "") -> bool:
    if obj_type in LANDMARK_TYPES:
        return True
    text = str(asset_or_id).lower()
    return any(name.lower() in text for name in LANDMARK_TYPES)


def salient_receptacles(objects: Sequence[Dict[str, Any]], room_id, limit=TOP_RECEPTACLES):
    """The largest non-openable receptacles in ``room_id``.

    Floor, wall, ceiling, room, and openable receptacles are excluded.
    """
    pool = []
    for obj in objects:
        if not obj.get("receptacle"):
            continue
        if obj.get("pickupable"):
            continue
        if obj.get("openable"):
            continue
        oid = obj.get("objectId") or obj.get("id")
        if is_structural_object(obj_type=obj.get("objectType"), object_id=oid):
            continue
        if str(oid or "").startswith("Floor|") or obj.get("objectType") == "Floor":
            continue
        if obj.get("room_id") != room_id:
            continue
        pool.append(obj)
    pool.sort(key=lambda obj: (-receptacle_salience(obj), str(obj.get("objectId") or obj.get("id") or "")))
    return pool[:limit]


def xyz_dict(point) -> Dict[str, float]:
    if isinstance(point, dict):
        return {"x": float(point["x"]), "y": float(point["y"]), "z": float(point["z"])}
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def floor_anchor_points(landmarks, spawns, radius_m=FLOOR_ANCHOR_RADIUS_M):
    """Closest floor spawn within ``radius_m`` of each landmark.

    ``landmarks`` items have ``landmark_id`` and ``position``.
    ``spawns`` are xyz dicts. One point per landmark.
    """
    anchors = []
    for landmark in landmarks:
        if not landmark.get("landmark_id") or not landmark.get("position"):
            continue
        if is_structural_object(
            obj_type=landmark.get("obj-type"), object_id=landmark.get("landmark_id")
        ):
            continue
        origin = xyz_dict(landmark["position"])
        nearest = None
        nearest_d = None
        for pos in spawns:
            point = xyz_dict(pos)
            distance = float(np.hypot(point["x"] - origin["x"], point["z"] - origin["z"]))
            if distance > radius_m:
                continue
            if nearest_d is None or distance < nearest_d:
                nearest = point
                nearest_d = distance
        if nearest is not None:
            anchors.append((landmark["landmark_id"], nearest))
    return anchors
