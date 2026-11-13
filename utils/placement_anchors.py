"""Receptacle and floor anchors for per-house placement trials."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from geometry import aabb_bounds
from utils.constants.object_constants import is_structural_object
from utils.data_generation_utils.navigation_utils import get_room_id_from_location
from utils.collection_config import load_collection_config

_PLACEMENT = load_collection_config()["placement"]
TOP_RECEPTACLES = int(_PLACEMENT["top_receptacles"])
TOP_LANDMARKS = int(_PLACEMENT["top_landmarks"])
FLOOR_CONTACT_M = 0.1
PASSAGE_TYPES = {"Door", "Doorway", "Doorframe"}
PASSAGE_ID_PREFIXES = ("door|", "doorway|", "doorframe|")


def room_id_at(room_poly_map, position):
    """Room that contains this xz position. Missing positions stay unset."""
    if not position:
        return None
    return get_room_id_from_location(room_poly_map, position, verbose=False)


def object_room_id(obj: Dict[str, Any], room_poly_map):
    """Room that contains the object.

    A ProcTHOR room floor has the room's id as its ``objectId`` and position
    (0, 0, 0), so its position says nothing about its room. The house-wide
    ``Floor`` object belongs to no room.
    """
    if _is_floor(obj):
        oid = obj.get("objectId") or obj.get("id")
        return oid if oid in room_poly_map else None
    return room_id_at(room_poly_map, obj.get("position"))


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


def _axis_y(value) -> Optional[float]:
    if isinstance(value, dict) and "y" in value:
        return float(value["y"])
    return None


def aabb_bottom_y(obj: Dict[str, Any]) -> Optional[float]:
    """Lowest y of the axis-aligned box."""
    box = obj.get("axisAlignedBoundingBox") or {}
    center_y = _axis_y(box.get("center"))
    size_y = _axis_y(box.get("size"))
    if center_y is not None and size_y is not None:
        return center_y - abs(size_y) / 2.0
    low, _high = aabb_bounds(box)
    if low is None:
        return None
    return low[1]


def floor_surface_y(obj: Dict[str, Any]) -> Optional[float]:
    """Top of the floor box, the walking surface."""
    box = obj.get("axisAlignedBoundingBox") or {}
    center_y = _axis_y(box.get("center"))
    size_y = _axis_y(box.get("size"))
    if center_y is not None and size_y is not None:
        return center_y + abs(size_y) / 2.0
    _low, high = aabb_bounds(box)
    if high is not None:
        return high[1]
    return _axis_y(obj.get("position"))


def is_passage_object(obj: Dict[str, Any], passage_ids=None) -> bool:
    """Doors and doorways listed as passages. They are not landmarks."""
    oid = str(obj.get("objectId") or obj.get("id") or "")
    if passage_ids and oid in passage_ids:
        return True
    if obj.get("objectType") in PASSAGE_TYPES:
        return True
    return oid.lower().startswith(PASSAGE_ID_PREFIXES)


def _is_floor(obj: Dict[str, Any]) -> bool:
    oid = obj.get("objectId") or obj.get("id")
    return obj.get("objectType") == "Floor" or str(oid or "").startswith("Floor|")


def floor_height_by_room(objects: Sequence[Dict[str, Any]]) -> Dict[Any, float]:
    grouped: Dict[Any, List[float]] = {}
    for obj in objects:
        if not _is_floor(obj):
            continue
        room = obj.get("room_id")
        height = floor_surface_y(obj)
        if room is None or height is None:
            continue
        grouped.setdefault(room, []).append(height)
    return {room: float(np.median(heights)) for room, heights in grouped.items()}


def salient_landmarks(
    objects: Sequence[Dict[str, Any]],
    room_id=None,
    limit: int = TOP_LANDMARKS,
    passage_ids=None,
) -> List[Dict[str, Any]]:
    """Largest non-pickupable, non-structural objects standing on the floor.

    An object qualifies when its box bottom is within ``FLOOR_CONTACT_M`` of
    that room's floor height. A room with no floor uses the lowest floor height
    in the house. Doors and doorways are excluded. Each room contributes at
    most ``limit`` objects. Objaverse instances qualify the same way.
    Type names are not used.
    """
    heights = floor_height_by_room(objects)
    fallback = min(heights.values()) if heights else None
    if room_id is not None:
        rooms = [room_id]
    else:
        rooms = sorted(
            {obj.get("room_id") for obj in objects if obj.get("room_id") is not None},
            key=str,
        )
    chosen: List[Dict[str, Any]] = []
    for room in rooms:
        floor_y = heights.get(room, fallback)
        if floor_y is None:
            continue
        pool = []
        for obj in objects:
            if obj.get("room_id") != room or obj.get("pickupable"):
                continue
            oid = obj.get("objectId") or obj.get("id")
            if not oid or is_structural_object(obj_type=obj.get("objectType"), object_id=oid):
                continue
            if is_passage_object(obj, passage_ids):
                continue
            bottom = aabb_bottom_y(obj)
            if bottom is None or abs(bottom - floor_y) > FLOOR_CONTACT_M:
                continue
            pool.append(obj)
        pool.sort(
            key=lambda obj: (
                -receptacle_salience(obj),
                str(obj.get("objectId") or obj.get("id") or ""),
            )
        )
        chosen.extend(pool[:limit])
    return chosen


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


def aabb_top_center(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Point at the center of the top face of the axis-aligned box."""
    box = obj.get("axisAlignedBoundingBox") or {}
    center = box.get("center")
    size = box.get("size")
    if isinstance(center, dict) and isinstance(size, dict) and "y" in center and "y" in size:
        return {
            "x": float(center["x"]),
            "y": float(center["y"]) + abs(float(size["y"])) / 2.0,
            "z": float(center["z"]),
        }
    low, high = aabb_bounds(box)
    if low is None or high is None:
        return None
    return {
        "x": (low[0] + high[0]) / 2.0,
        "y": high[1],
        "z": (low[2] + high[2]) / 2.0,
    }


def nearest_points(points, target: Dict[str, float], limit: int = 3) -> List[Dict[str, float]]:
    """Up to ``limit`` points closest to ``target``."""
    ranked = []
    for point in points:
        xyz = xyz_dict(point)
        distance = float(
            np.linalg.norm(
                [
                    xyz["x"] - target["x"],
                    xyz["y"] - target["y"],
                    xyz["z"] - target["z"],
                ]
            )
        )
        ranked.append((distance, xyz))
    ranked.sort(key=lambda item: (item[0], item[1]["x"], item[1]["y"], item[1]["z"]))
    return [point for _distance, point in ranked[:limit]]


def spread_points(
    points, target: Dict[str, float], limit: int, min_separation_m: float
) -> List[Dict[str, float]]:
    """Up to ``limit`` points, closest to ``target`` first, at least
    ``min_separation_m`` apart on the floor plane."""
    chosen: List[Dict[str, float]] = []
    for point in nearest_points(points, target, len(points)):
        if all(
            np.hypot(point["x"] - q["x"], point["z"] - q["z"]) >= min_separation_m
            for q in chosen
        ):
            chosen.append(point)
            if len(chosen) == limit:
                break
    return chosen


def place_lift(obj: Dict[str, Any], epsilon_m: float) -> float:
    """Height from a surface point to where the object's position must go.

    PlaceObjectAtPoint puts the object's position (pivot) at the target, so a
    surface point must be raised by pivot-to-bottom height, plus a small gap.
    """
    box = obj.get("axisAlignedBoundingBox") or {}
    center, size = box.get("center"), box.get("size")
    position = obj.get("position")
    if not (isinstance(center, dict) and isinstance(size, dict) and position):
        return float(epsilon_m)
    bottom = float(center["y"]) - abs(float(size["y"])) / 2.0
    return max(0.0, float(position["y"]) - bottom) + float(epsilon_m)


def lifted(point, lift: float) -> Dict[str, float]:
    xyz = xyz_dict(point)
    xyz["y"] += lift
    return xyz


def xyz_dict(point) -> Dict[str, float]:
    if isinstance(point, dict):
        return {"x": float(point["x"]), "y": float(point["y"]), "z": float(point["z"])}
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def footprint_half_extent(obj: Dict[str, Any]) -> float:
    """Half of the longer horizontal side of the axis-aligned box (0 without a box)."""
    size = (obj.get("axisAlignedBoundingBox") or {}).get("size")
    if not isinstance(size, dict):
        return 0.0
    return max(abs(float(size.get("x", 0.0))), abs(float(size.get("z", 0.0)))) / 2.0


def footprint_gap(point: Dict[str, float], landmark: Dict[str, Any]) -> float:
    """Horizontal distance from ``point`` to the landmark's box footprint (0 inside).

    A landmark without a ``box`` is measured from its ``position``.
    """
    box = landmark.get("box") or {}
    center, size = box.get("center"), box.get("size")
    if isinstance(center, dict) and isinstance(size, dict):
        dx = abs(point["x"] - float(center["x"])) - abs(float(size["x"])) / 2.0
        dz = abs(point["z"] - float(center["z"])) - abs(float(size["z"])) / 2.0
        return float(np.hypot(max(dx, 0.0), max(dz, 0.0)))
    origin = xyz_dict(landmark["position"])
    return float(np.hypot(point["x"] - origin["x"], point["z"] - origin["z"]))


def floor_anchor_points(
    landmarks,
    spawns,
    radius_m: float,
    clearance_m: float = 0.0,
    limit: int = 1,
    min_separation_m: float = 0.0,
):
    """Floor spawns next to each landmark, nearest first.

    ``landmarks`` items have ``landmark_id``, ``position`` and optionally ``box``
    (the axis-aligned box). ``spawns`` are xyz dicts. A spawn qualifies when its
    gap to the landmark footprint is between ``clearance_m`` (room for the placed
    object) and ``radius_m``. Each landmark gives up to ``limit`` points at least
    ``min_separation_m`` apart, as ``(landmark_id, point)`` pairs.
    """
    points = [xyz_dict(pos) for pos in spawns]
    anchors = []
    for landmark in landmarks:
        if not landmark.get("landmark_id") or not landmark.get("position"):
            continue
        if is_structural_object(
            obj_type=landmark.get("obj-type"), object_id=landmark.get("landmark_id")
        ):
            continue
        ranked = []
        for point in points:
            gap = footprint_gap(point, landmark)
            if clearance_m <= gap <= radius_m:
                ranked.append((gap, point["x"], point["z"], point))
        ranked.sort(key=lambda item: item[:3])
        chosen: List[Dict[str, float]] = []
        for _gap, _x, _z, point in ranked:
            if all(
                np.hypot(point["x"] - q["x"], point["z"] - q["z"]) >= min_separation_m
                for q in chosen
            ):
                chosen.append(point)
                if len(chosen) == limit:
                    break
        anchors.extend((landmark["landmark_id"], point) for point in chosen)
    return anchors


def stands_on_floor(obj: Dict[str, Any], floor_y: Optional[float]) -> bool:
    """True when the box bottom is within ``FLOOR_CONTACT_M`` of the floor height."""
    bottom = aabb_bottom_y(obj)
    return floor_y is not None and bottom is not None and abs(bottom - floor_y) <= FLOOR_CONTACT_M


def is_trial_object(obj: Dict[str, Any], include_moveable: bool, max_mass_kg: float) -> bool:
    """Objects whose placement is tried: every pickupable, and (when
    ``include_moveable``) a ``moveable`` object that is light enough and has
    nothing resting on it, so moving it leaves the rest of the house in place.
    Doors and structure are never moved.
    """
    oid = obj.get("objectId")
    if not oid:
        return False
    if obj.get("pickupable"):
        return True
    if not include_moveable or not obj.get("moveable"):
        return False
    if is_structural_object(obj_type=obj.get("objectType"), object_id=oid):
        return False
    if is_passage_object(obj):
        return False
    if obj.get("receptacleObjectIds"):
        return False
    mass = obj.get("mass")
    return mass is not None and float(mass) <= float(max_mass_kg)
