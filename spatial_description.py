"""
spatial_description.py

Utilities for generating natural language spatial descriptions
from 3D positions, camera pose, and (optionally) 3D bounding boxes.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CameraPose:
    """
    Camera extrinsics: R (3x3) and t (3,)
    We assume world points p_world are transformed to camera/view coords by:
        p_view = R @ (p_world - t)
    """
    R: np.ndarray  # shape (3, 3)
    t: np.ndarray  # shape (3,)


@dataclass
class Object3D:
    name: str
    position: np.ndarray          # (3,) world coordinates (center of object)
    bbox_size: Optional[np.ndarray] = None  # (3,) [width, height, depth] in meters
    orientation_R: Optional[np.ndarray] = None  # (3,3) rotation matrix of object in world frame


# -----------------------------
# Basic geometry helpers
# -----------------------------

def world_to_view(p_world: np.ndarray, cam: CameraPose) -> np.ndarray:
    """Transform world coordinates to camera/view coordinates."""
    return cam.R @ (p_world - cam.t)


def direction_label_from_view_pos(
    p_view: np.ndarray,
    tau_lat: float = 0.2,
    tau_depth: float = 0.2,
    tau_vert: float = 0.2,
) -> Dict[str, str]:
    """
    Convert a 3D point in view coordinates into discrete labels:
    - depth: in front / behind
    - horiz: left / right / centered
    - vert: above / below / level
    """
    x, y, z = p_view

    # depth
    if z > tau_depth:
        depth = "in front of you"
    elif z < -tau_depth:
        depth = "behind you"
    else:
        depth = "at your position"

    # horizontal
    if x > tau_lat:
        horiz = "to your right"
    elif x < -tau_lat:
        horiz = "to your left"
    else:
        horiz = "roughly centered"

    # vertical
    if y > tau_vert:
        vert = "above your eye level"
    elif y < -tau_vert:
        vert = "below your eye level"
    else:
        vert = "around your eye level"

    return {"depth": depth, "horiz": horiz, "vert": vert}


def format_single_object_position_sentence(name: str, labels: Dict[str, str]) -> str:
    depth = labels["depth"]
    horiz = labels["horiz"]
    vert = labels["vert"]

    # Slightly smarter phrasing
    parts = [depth]
    if horiz != "roughly centered":
        parts.append(horiz)
    if vert != "around your eye level":
        parts.append(vert)

    # Fallback if everything is neutral
    if not parts:
        parts = ["near you"]

    relation_phrase = ", ".join(parts)
    return f"The {name} is {relation_phrase}."


def pairwise_relation_from_view_positions(
    pA: np.ndarray,
    pB: np.ndarray,
    nameA: str,
    nameB: str,
    tau_lat: float = 0.2,
    tau_depth: float = 0.2,
    tau_vert: float = 0.2,
) -> str:
    """
    Very simple pairwise relation: left/right, front/behind, above/below of A relative to B,
    all in the camera/view frame.
    """
    dx = pA[0] - pB[0]
    dy = pA[1] - pB[1]
    dz = pA[2] - pB[2]

    horiz = ""
    if dx > tau_lat:
        horiz = "to the right of"
    elif dx < -tau_lat:
        horiz = "to the left of"

    depth = ""
    if dz > tau_depth:
        depth = "behind"
    elif dz < -tau_depth:
        depth = "in front of"

    vert = ""
    if dy > tau_vert:
        vert = "above"
    elif dy < -tau_vert:
        vert = "below"

    components = []
    if horiz:
        components.append(horiz)
    if depth:
        components.append(depth)
    if vert:
        components.append(vert)

    if not components:
        return f"The {nameA} is roughly aligned with the {nameB}."

    phrase = " and ".join(components)
    return f"The {nameA} is {phrase} the {nameB}."


# -----------------------------
# 3D bounding box helpers
# -----------------------------

def get_bbox_corners(obj: Object3D) -> Optional[np.ndarray]:
    """
    Return 8 corners of the object's 3D bounding box in world coordinates.
    If bbox_size is None, returns None.

    bbox_size = [width, height, depth].
    We assume the box is centered at obj.position and oriented by obj.orientation_R
    (or axis-aligned if orientation_R is None).
    """
    if obj.bbox_size is None:
        return None

    w, h, d = obj.bbox_size
    cx, cy, cz = obj.position

    # local offsets of corners (axis-aligned in object local frame)
    half = np.array([w / 2.0, h / 2.0, d / 2.0])
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    local_corners = signs * half  # shape (8,3)

    if obj.orientation_R is None:
        # world-aligned
        world_corners = local_corners + obj.position
    else:
        # rotate by object orientation
        world_corners = (obj.orientation_R @ local_corners.T).T + obj.position

    return world_corners  # (8,3)


def bbox_extents_in_view(
    corners_world: np.ndarray,
    cam: CameraPose,
) -> Dict[str, float]:
    """
    Given bbox corners in world coords, compute min/max extents in view coords.
    Return dict with xmin, xmax, ymin, ymax, zmin, zmax.
    """
    corners_view = np.array([world_to_view(c, cam) for c in corners_world])
    xs = corners_view[:, 0]
    ys = corners_view[:, 1]
    zs = corners_view[:, 2]

    return {
        "xmin": xs.min(),
        "xmax": xs.max(),
        "ymin": ys.min(),
        "ymax": ys.max(),
        "zmin": zs.min(),
        "zmax": zs.max(),
    }


def boxes_overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    """Check if 1D intervals overlap."""
    return not (a_max < b_min or b_max < a_min)


def detect_support_relation(
    extA: Dict[str, float], extB: Dict[str, float], tol_height: float = 0.05
) -> Optional[str]:
    """
    Simple heuristic: A 'on top of' B if:
      - bottom of A is approx top of B (within tol_height)
      - and they overlap in x and z in view space.

    NOTE: This is view-space based; you can instead do this in world-space if you prefer.
    """
    # bottom/top in vertical axis (y)
    bottomA = extA["ymin"]
    topB = extB["ymax"]

    if abs(bottomA - topB) > tol_height:
        return None

    overlap_x = boxes_overlap_1d(extA["xmin"], extA["xmax"], extB["xmin"], extB["xmax"])
    overlap_z = boxes_overlap_1d(extA["zmin"], extA["zmax"], extB["zmax"], extB["zmin"])

    if overlap_x and overlap_z:
        return "on_top"

    return None


def detect_occlusion_relation(
    extFront: Dict[str, float],
    extBack: Dict[str, float],
    depth_margin: float = 0.1,
) -> bool:
    """
    Very rough heuristic: 'Front' is in front of 'Back' and overlaps in x,y.
    """
    if extFront["zmax"] + depth_margin < extBack["zmin"]:
        # front box entirely closer than back box
        overlap_x = boxes_overlap_1d(
            extFront["xmin"], extFront["xmax"], extBack["xmin"], extBack["xmax"]
        )
        overlap_y = boxes_overlap_1d(
            extFront["ymin"], extFront["ymax"], extBack["ymin"], extBack["ymax"]
        )
        return overlap_x and overlap_y
    return False


# -----------------------------
# Public API: positions only
# -----------------------------

def describe_scene_positions_only(
    objects: List[Object3D], cam: CameraPose
) -> Dict[str, Any]:
    """
    Generate a simple description using only 3D positions and camera pose.
    Returns a dict with:
      - "per_object": list of sentences
      - "pairwise": list of sentences
      - "summary": one paragraph string
    """
    if not objects:
        return {
            "per_object": [],
            "pairwise": [],
            "summary": "There are no objects in the scene."
        }

    # Compute view-space positions
    view_positions = {
        obj.name: world_to_view(obj.position, cam) for obj in objects
    }

    # Per-object sentences
    per_object_sentences = []
    for obj in objects:
        p_view = view_positions[obj.name]
        labels = direction_label_from_view_pos(p_view)
        sent = format_single_object_position_sentence(obj.name, labels)
        per_object_sentences.append(sent)

    # Simple pairwise sentences (can be pruned later)
    pairwise_sentences = []
    for i, objA in enumerate(objects):
        for j, objB in enumerate(objects):
            if i >= j:
                continue
            pA = view_positions[objA.name]
            pB = view_positions[objB.name]
            sentAB = pairwise_relation_from_view_positions(pA, pB, objA.name, objB.name)
            sentBA = pairwise_relation_from_view_positions(pB, pA, objB.name, objA.name)
            pairwise_sentences.extend([sentAB, sentBA])

    # Simple summary: anchor on closest object in depth
    depths = [(name, vp[2]) for name, vp in view_positions.items()]
    depths_sorted = sorted(depths, key=lambda x: x[1])
    closest_name = depths_sorted[0][0]

    summary = (
        f"From your viewpoint, the {closest_name} is the closest object. "
        f"Other objects are distributed around it as described: "
        + " ".join(per_object_sentences)
    )

    return {
        "per_object": per_object_sentences,
        "pairwise": pairwise_sentences,
        "summary": summary,
    }


# -----------------------------
# Public API: with 3D bounding boxes
# -----------------------------

def describe_scene_with_bboxes(
    objects: List[Object3D], cam: CameraPose
) -> Dict[str, Any]:
    """
    Generate a richer description using positions + 3D bounding boxes.
    Returns a dict with:
      - "per_object": list of sentences
      - "pairwise": list of sentences (including support/occlusion)
      - "summary": one paragraph string
    """
    if not objects:
        return {
            "per_object": [],
            "pairwise": [],
            "summary": "There are no objects in the scene."
        }

    view_positions = {
        obj.name: world_to_view(obj.position, cam) for obj in objects
    }

    bbox_extents = {}
    for obj in objects:
        corners = get_bbox_corners(obj)
        if corners is not None:
            bbox_extents[obj.name] = bbox_extents_in_view(corners, cam)

    # Per-object sentences: position + size if bbox available
    per_object_sentences = []
    for obj in objects:
        p_view = view_positions[obj.name]
        labels = direction_label_from_view_pos(p_view)
        base_sent = format_single_object_position_sentence(obj.name, labels)

        if obj.bbox_size is not None:
            w, h, d = obj.bbox_size
            size_desc = f"It spans roughly {w:.2f} m wide, {h:.2f} m tall, and {d:.2f} m deep."
            base_sent = base_sent.rstrip(".") + f". {size_desc}"

        per_object_sentences.append(base_sent)

    # Pairwise: positions + support + occlusion
    pairwise_sentences = []
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            A = objects[i]
            B = objects[j]
            pA = view_positions[A.name]
            pB = view_positions[B.name]

            # Position-based relation
            pos_sent = pairwise_relation_from_view_positions(pA, pB, A.name, B.name)
            pairwise_sentences.append(pos_sent)

            # Support (A on top of B or B on top of A)
            extA = bbox_extents.get(A.name)
            extB = bbox_extents.get(B.name)
            if extA is not None and extB is not None:
                supAB = detect_support_relation(extA, extB)
                supBA = detect_support_relation(extB, extA)

                if supAB == "on_top":
                    pairwise_sentences.append(f"The {A.name} appears to rest on top of the {B.name}.")
                if supBA == "on_top":
                    pairwise_sentences.append(f"The {B.name} appears to rest on top of the {A.name}.")

                # Occlusion: whichever is closer is potentially occluding the other
                if detect_occlusion_relation(extA, extB):
                    pairwise_sentences.append(f"From your viewpoint, the {A.name} partially occludes the {B.name}.")
                if detect_occlusion_relation(extB, extA):
                    pairwise_sentences.append(f"From your viewpoint, the {B.name} partially occludes the {A.name}.")

    # Summary: choose a large "anchor" object (by bbox volume) if possible
    anchor_name = None
    max_volume = -1.0
    for obj in objects:
        if obj.bbox_size is not None:
            w, h, d = obj.bbox_size
            vol = w * h * d
            if vol > max_volume:
                max_volume = vol
                anchor_name = obj.name

    if anchor_name is None:
        # fallback to closest in depth
        depths = [(name, vp[2]) for name, vp in view_positions.items()]
        depths_sorted = sorted(depths, key=lambda x: x[1])
        anchor_name = depths_sorted[0][0]

    summary = (
        f"In this scene, the {anchor_name} acts as a central reference object. "
        f"Other objects are arranged around it as follows: "
        + " ".join(per_object_sentences)
    )

    return {
        "per_object": per_object_sentences,
        "pairwise": pairwise_sentences,
        "summary": summary,
    }


# -----------------------------
# Example usage (remove in production)
# -----------------------------
if __name__ == "__main__":
    # Example camera: identity pose at origin
    cam = CameraPose(R=np.eye(3), t=np.zeros(3))

    # Simple toy scene
    objs = [
        Object3D(name="table", position=np.array([0.0, 0.0, 3.0]), bbox_size=np.array([1.5, 0.8, 0.9])),
        Object3D(name="chair", position=np.array([-1.0, 0.0, 2.5]), bbox_size=np.array([0.6, 1.0, 0.6])),
        Object3D(name="laptop", position=np.array([0.2, 0.7, 3.1]), bbox_size=np.array([0.3, 0.2, 0.2])),
    ]

    desc_pos = describe_scene_positions_only(objs, cam)
    print("=== Positions only ===")
    print(desc_pos["summary"])
    for s in desc_pos["per_object"]:
        print("-", s)

    # desc_bbox = describe_scene_with_bboxes(objs, cam)
    # print("\n=== With bboxes ===")
    # print(desc_bbox["summary"])
    # for s in desc_bbox["per_object"]:
    #     print("-", s)
