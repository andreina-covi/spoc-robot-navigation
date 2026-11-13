"""Open-loop glance at nearby movable objects during RoomVisit collection.

The targets are the objects the placement trials move (``is_trial_object``):
every pickupable and, with ``placement.include_moveable``, light ``moveable``
objects such as chairs, plants and lamps. Object positions and sizes are read
once at episode start (nothing moves during RoomVisit). Each step costs one
vectorised distance/bearing computation:
no rendering, no extra simulator calls. A glance is a fixed number of 45° turns
computed once; nothing loops until the object is visible.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from utils.placement_anchors import is_trial_object


@dataclass(frozen=True)
class GlanceSettings:
    min_frame_fraction: float
    area_margin: float
    max_distance_m: float
    max_turns: int
    rotate_back: bool
    max_per_episode: int
    min_steps_between: int

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GlanceSettings":
        return cls(
            min_frame_fraction=float(cfg["min_frame_fraction"]),
            area_margin=float(cfg["area_margin"]),
            max_distance_m=float(cfg["max_distance_m"]),
            max_turns=int(cfg["max_turns"]),
            rotate_back=bool(cfg["rotate_back"]),
            max_per_episode=int(cfg["max_per_episode"]),
            min_steps_between=int(cfg["min_steps_between"]),
        )


@dataclass(frozen=True)
class GlanceTarget:
    object_id: str
    turns: int  # signed: > 0 turn right, < 0 turn left
    bearing_deg: float
    distance_m: float
    expected_px: float


def horizontal_fov_deg(width: int, height: int, vertical_fov_deg: float) -> float:
    return 2 * math.degrees(
        math.atan(width / height * math.tan(math.radians(vertical_fov_deg / 2)))
    )


def is_hidden_in_closed_receptacle(obj: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> bool:
    for pid in obj.get("parentReceptacles") or []:
        parent = by_id.get(pid) or {}
        if parent.get("openable", False) and not parent.get("isOpen", False):
            return True
    return False


def glance_candidates(
    objects: Iterable[Dict[str, Any]], include_moveable: bool, max_mass_kg: float
) -> List[Dict[str, Any]]:
    """THOR objects worth a glance: the placement-trial objects that can be seen.

    Same set as ``scripts/placement_trials.py`` moves, so every object with
    placement candidates can also get a clear sighting. An object inside a closed
    receptacle or without a bounding box is left out.
    """
    objects = list(objects)
    by_id = {o["objectId"]: o for o in objects if o.get("objectId")}
    kept = []
    for o in objects:
        if not is_trial_object(o, include_moveable, max_mass_kg):
            continue
        if is_hidden_in_closed_receptacle(o, by_id):
            continue
        box = o.get("axisAlignedBoundingBox") or {}
        if not box.get("size") or not (box.get("center") or o.get("position")):
            continue
        kept.append(o)
    return kept


class GlancePlanner:
    """Chooses at most one glance target per policy step."""

    def __init__(
        self,
        objects: Iterable[Dict[str, Any]],
        settings: GlanceSettings,
        frame_width: int,
        frame_height: int,
        vertical_fov_deg: float,
        rotation_deg: float,
    ) -> None:
        """``objects``: dicts with id, x, z, width (horizontal extent, m),
        height (m) and room (room id or None)."""
        self.settings = settings
        self.rotation_deg = float(rotation_deg)
        self.half_hfov = horizontal_fov_deg(frame_width, frame_height, vertical_fov_deg) / 2
        self.fx = frame_width / 2 / math.tan(math.radians(self.half_hfov))
        self.target_px = settings.area_margin * settings.min_frame_fraction * frame_width * frame_height
        # Largest bearing that max_turns brings within half a turn of the centre.
        self.max_bearing = (settings.max_turns + 0.5) * self.rotation_deg

        objects = list(objects)
        self.ids: List[str] = [o["id"] for o in objects]
        self.rooms = np.array([o.get("room") for o in objects], dtype=object)
        self.xz = np.array([[o["x"], o["z"]] for o in objects], dtype=float).reshape(-1, 2)
        self.face_area = np.array(
            [max(o["width"], 0.0) * max(o["height"], 0.0) for o in objects], dtype=float
        )
        self.glanced: Set[str] = set()
        self.n_glances = 0
        self.last_glance_step: Optional[int] = None

    def turns_for(self, bearing_deg: float) -> int:
        n = int(round(abs(bearing_deg) / self.rotation_deg))
        n = max(1, min(n, self.settings.max_turns))
        return n if bearing_deg > 0 else -n

    def pick(
        self,
        agent_x: float,
        agent_z: float,
        yaw_deg: float,
        room: Optional[str],
        step: int,
    ) -> Optional[GlanceTarget]:
        s = self.settings
        if not len(self.ids) or self.n_glances >= s.max_per_episode:
            return None
        if self.last_glance_step is not None and step - self.last_glance_step < s.min_steps_between:
            return None

        d = self.xz - np.array([agent_x, agent_z])
        dist = np.hypot(d[:, 0], d[:, 1])
        yaw = math.radians(yaw_deg)
        fwd = d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
        right = d[:, 0] * math.cos(yaw) - d[:, 1] * math.sin(yaw)
        bearing = np.degrees(np.arctan2(right, fwd))
        expected_px = self.face_area * (self.fx / np.maximum(dist, 1e-3)) ** 2

        ok = (
            (dist <= s.max_distance_m)
            & (expected_px >= self.target_px)
            & (np.abs(bearing) > self.half_hfov)
            & (np.abs(bearing) <= self.max_bearing)
        )
        if room is not None:
            ok &= self.rooms == room
        if self.glanced:
            ok &= np.array([oid not in self.glanced for oid in self.ids])
        if not ok.any():
            return None

        i = int(np.argmax(np.where(ok, expected_px, -1.0)))
        return GlanceTarget(
            object_id=self.ids[i],
            turns=self.turns_for(float(bearing[i])),
            bearing_deg=float(bearing[i]),
            distance_m=float(dist[i]),
            expected_px=float(expected_px[i]),
        )

    def mark_done(self, target: GlanceTarget, step: int) -> None:
        self.glanced.add(target.object_id)
        self.n_glances += 1
        self.last_glance_step = step
