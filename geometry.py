"""Project THOR 3D bounding-box corners to expected 2D image bbox / angular size.

Uses camera intrinsics from vertical FOV and Unity/AI2-THOR yaw+pitch extrinsics.
Occlusion is ignored: this is the unoccluded expected footprint on the image plane.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

Point3 = Union[Sequence[float], Dict[str, float], np.ndarray]

# Fallback if StretchController has not calibrated yet. Prefer the live value from
# ``StretchController.nav_camera_mount_deg`` (set in ``calibrate_agent``).
DEFAULT_NAV_CAMERA_MOUNT_DEG = 27.0


def camera_intrinsics(W: float, H: float, fov_v_deg: float) -> Tuple[float, float]:
    """Focal length in pixels (square pixels) and horizontal FOV in degrees."""
    fov_v = math.radians(fov_v_deg)
    f_px = (H / 2.0) / math.tan(fov_v / 2.0)
    fov_h = 2.0 * math.atan((W / 2.0) / f_px)
    return f_px, math.degrees(fov_h)


def _as_xyz(p: Point3) -> np.ndarray:
    if isinstance(p, dict):
        return np.array(
            [float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64
        )
    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"Expected xyz, got {p!r}")
    return arr[:3].copy()


def _ang(p1: float, p2: float, center: float, f: float) -> float:
    return math.degrees(
        abs(math.atan((p2 - center) / f) - math.atan((p1 - center) / f))
    )


def angular_size_deg(
    cmin: float,
    cmax: float,
    rmin: float,
    rmax: float,
    W: float,
    H: float,
    f_px: float,
) -> Tuple[float, float]:
    """Angular width/height (degrees) subtended by a pixel bbox, given focal length ``f_px``."""
    cx, cy = W / 2.0, H / 2.0
    width_deg = _ang(cmin, cmax, cx, f_px)
    height_deg = _ang(rmin, rmax, cy, f_px)
    return width_deg, height_deg


def yaw_pitch_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World→camera rotation for AI2-THOR / Unity (Y-up, agent forward from yaw).

    Agent forward at yaw ``y`` is ``(sin y, 0, cos y)``. Camera +Z is forward,
    +X right, +Y up. THOR ``cameraHorizon`` / mount pitch are **look-down positive**,
    so pitch is applied with a negative sign in the math convention.
    """
    yaw = math.radians(yaw_deg)
    # THOR look-down-positive → math pitch (look-up-positive)
    pitch = math.radians(-pitch_deg)

    # Rows: camera right, up, forward in world coordinates
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array(
        [
            [cy, 0.0, -sy],
            [0.0, 1.0, 0.0],
            [sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )
    return R_pitch @ R_yaw


def project_points(
    points_world: Iterable[Point3],
    cam_pos: Point3,
    yaw_deg: float,
    pitch_deg: float,
    W: float,
    H: float,
    f_px: float,
) -> List[Optional[Tuple[float, float]]]:
    """Project world points to (col, row). Points behind the camera → ``None``.

    Image row increases downward; world/camera Y is up, so
    ``row = cy - f * (y_cam / z_cam)``.
    """
    R = yaw_pitch_rotation_matrix(yaw_deg, pitch_deg)
    cam = _as_xyz(cam_pos)
    cx, cy = W / 2.0, H / 2.0

    out: List[Optional[Tuple[float, float]]] = []
    for p in points_world:
        p_cam = R @ (_as_xyz(p) - cam)
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        if z <= 1e-4:
            out.append(None)
            continue
        col = cx + f_px * (x / z)
        row = cy - f_px * (y / z)
        out.append((col, row))
    return out


def corners_from_center_size(center: Point3, size: Point3) -> List[np.ndarray]:
    """Axis-aligned 8 corners from THOR ``center`` + ``size``."""
    c = _as_xyz(center)
    s = _as_xyz(size)
    half = np.abs(s) * 0.5
    corners = []
    for dx in (-half[0], half[0]):
        for dy in (-half[1], half[1]):
            for dz in (-half[2], half[2]):
                corners.append(c + np.array([dx, dy, dz], dtype=np.float64))
    return corners


def corners_from_box(box: Any) -> Optional[List[np.ndarray]]:
    """Extract 8 world corners from a THOR AABB/OOBB dict (or a raw corner list)."""
    if box is None:
        return None

    if isinstance(box, dict):
        corners = box.get("cornerPoints")
        if corners:
            try:
                return [_as_xyz(c) for c in corners]
            except Exception:
                pass
        center = box.get("center")
        size = box.get("size")
        if center is not None and size is not None:
            try:
                return corners_from_center_size(center, size)
            except Exception:
                return None
        return None

    try:
        return [_as_xyz(c) for c in box]
    except Exception:
        return None


def box_center_size(aabb: Any = None, oobb: Any = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """World-space center and size for objects CSV (3D metadata, not segmentation).

    Prefer AABB ``center``/``size`` (always present in THOR). OOBB often has
    ``cornerPoints`` only — falling back to OOBB ``.get('center', 0)`` yields zeros.
    If needed, synthesize from cornerPoints.
    """
    for box in (aabb, oobb):
        if not isinstance(box, dict):
            continue
        center = box.get("center")
        size = box.get("size")
        if center is None or size is None:
            continue
        try:
            c = _as_xyz(center)
            s = _as_xyz(size)
        except Exception:
            continue
        if float(np.linalg.norm(s)) <= 1e-8:
            continue
        return (
            {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
            {"x": float(s[0]), "y": float(s[1]), "z": float(s[2])},
        )

    corners = corners_from_box(oobb) or corners_from_box(aabb)
    if corners:
        arr = np.stack(corners, axis=0)
        c = arr.mean(axis=0)
        s = arr.max(axis=0) - arr.min(axis=0)
        return (
            {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
            {"x": float(s[0]), "y": float(s[1]), "z": float(s[2])},
        )
    return {"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 0.0}


def expected_bbox_from_3d(
    obb_corners_world: Any,
    cam_pos: Point3,
    yaw_deg: float,
    pitch_deg: float,
    W: float,
    H: float,
    f_px: float,
    *,
    clamp_to_image: bool = True,
    min_depth: float = 0.05,
) -> Optional[Dict[str, float]]:
    """Unoccluded 2D bbox from 3D box corners.

    ``clamp_to_image=True`` (default) reports the on-screen expected footprint so
    area stays comparable to the frame. Corners closer than ``min_depth`` in camera
    space are ignored to avoid perspective blow-ups near the image plane.
    """
    corners = corners_from_box(obb_corners_world)
    if not corners:
        return None

    R = yaw_pitch_rotation_matrix(yaw_deg, pitch_deg)
    cam = _as_xyz(cam_pos)
    cx, cy = W / 2.0, H / 2.0

    cols: List[float] = []
    rows: List[float] = []
    for p in corners:
        p_cam = R @ (_as_xyz(p) - cam)
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        if z < min_depth:
            continue
        cols.append(cx + f_px * (x / z))
        rows.append(cy - f_px * (y / z))

    if not cols:
        return None

    cmin, cmax = min(cols), max(cols)
    rmin, rmax = min(rows), max(rows)
    area_raw = (cmax - cmin) * (rmax - rmin)

    if clamp_to_image:
        cmin_c = max(0.0, cmin)
        cmax_c = min(float(W), cmax)
        rmin_c = max(0.0, rmin)
        rmax_c = min(float(H), rmax)
        if cmax_c <= cmin_c or rmax_c <= rmin_c:
            return None
        return dict(
            cmin=cmin_c,
            rmin=rmin_c,
            cmax=cmax_c,
            rmax=rmax_c,
            area=(cmax_c - cmin_c) * (rmax_c - rmin_c),
            area_raw=area_raw,
        )

    if cmax <= cmin or rmax <= rmin:
        return None
    return dict(
        cmin=cmin,
        rmin=rmin,
        cmax=cmax,
        rmax=rmax,
        area=area_raw,
        area_raw=area_raw,
    )
