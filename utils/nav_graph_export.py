"""Build AI2-THOR reachability nav-graph exports (independent of the agent trajectory)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Defaults match utils.constants.stretch_initialization_utils (avoid importing ai2thor here).
_DEFAULT_AGENT_MOVE_M = 0.2
_DEFAULT_AGENT_ROTATION_DEG = 45.0


def reachability_grid_size(move_m: float = _DEFAULT_AGENT_MOVE_M) -> float:
    """Grid size used by StretchController.get_reachable_positions (default)."""
    return float(move_m) * 0.75


def _xz(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["z"])
    return float(p[0]), float(p[2])


def _y(p: Any) -> float:
    if isinstance(p, dict):
        return float(p.get("y", 0.0))
    return float(p[1]) if len(p) > 1 else 0.0


def build_nav_graph_from_reachable_positions(
    positions: Sequence[Any],
    *,
    grid_size: Optional[float] = None,
    agent_move_m: float = _DEFAULT_AGENT_MOVE_M,
    agent_rotation_deg: float = _DEFAULT_AGENT_ROTATION_DEG,
    connectivity: str = "8",
    snapshot: str = "episode_start",
    scene_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    door_states: Optional[List[Dict[str, Any]]] = None,
    extra_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Build nodes + edges from ``GetReachablePositions`` results.

    Edges are **not** Euclidean-nearest-neighbor. They are grid-adjacency on the
    reachability sample returned by THOR:

    - ``connectivity='4'``: axis-aligned neighbors at ~1 grid cell
    - ``connectivity='8'``: also diagonal neighbors (default)

    SPOC movement uses ``agent_move_m`` (typically 0.2 m) while the reachability
    sample uses ``grid_size`` (typically ``0.75 * agent_move_m``). Consumers that
    want action-length edges should use ``agent_move_m`` / trajectory actions;
    the graph here describes standable cells and local THOR reachability adjacency.
    """
    gs = float(grid_size) if grid_size is not None else reachability_grid_size(agent_move_m)
    if gs <= 0:
        raise ValueError(f"grid_size must be positive, got {gs}")

    # Stable node order: sort by (x, z, y) then assign ids
    raw = []
    for p in positions or []:
        x, z = _xz(p)
        y = _y(p)
        raw.append((round(x, 4), round(y, 4), round(z, 4)))
    raw = sorted(set(raw))

    nodes: List[Dict[str, Any]] = []
    key_to_id: Dict[Tuple[float, float, float], str] = {}
    for i, (x, y, z) in enumerate(raw):
        nid = f"n{i}"
        key_to_id[(x, y, z)] = nid
        nodes.append({"node_id": nid, "x": x, "y": y, "z": z})

    # Quantize to grid indices for adjacency (relative to min corner)
    if raw:
        xs = [t[0] for t in raw]
        zs = [t[2] for t in raw]
        x0, z0 = min(xs), min(zs)
    else:
        x0 = z0 = 0.0

    def cell(x: float, z: float) -> Tuple[int, int]:
        return (int(round((x - x0) / gs)), int(round((z - z0) / gs)))

    cell_to_keys: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    for key in raw:
        c = cell(key[0], key[2])
        cell_to_keys.setdefault(c, []).append(key)

    if connectivity == "4":
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif connectivity == "8":
        offsets = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    else:
        raise ValueError("connectivity must be '4' or '8'")

    # Max distance for a single grid step (diagonal for 8-conn)
    max_step = gs * (np.sqrt(2.0) if connectivity == "8" else 1.0) * 1.05

    edges: List[Dict[str, Any]] = []
    seen_pairs = set()
    for (x, y, z), nid in key_to_id.items():
        ci, cj = cell(x, z)
        for di, dj in offsets:
            for other in cell_to_keys.get((ci + di, cj + dj), []):
                ox, oy, oz = other
                # Prefer same floor height band
                if abs(oy - y) > 0.35:
                    continue
                dist = float(np.hypot(ox - x, oz - z))
                if dist <= 1e-6 or dist > max_step:
                    continue
                oid = key_to_id[other]
                a, b = (nid, oid) if nid < oid else (oid, nid)
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                edges.append(
                    {
                        "from_node": a,
                        "to_node": b,
                        "distance_xz": round(dist, 4),
                        "cost": round(dist, 4),
                        "bidirectional": True,
                    }
                )

    door_snapshot = None
    if door_states is not None:
        door_snapshot = [
            {
                "door_id": d.get("door_id") or d.get("passage_id"),
                "is_open": d.get("is_open"),
                "openness": d.get("openness"),
                "room0": d.get("room0") or d.get("from_region"),
                "room1": d.get("room1") or d.get("to_region"),
            }
            for d in door_states
        ]

    notes = (
        "Nodes from AI2-THOR GetReachablePositions. Edges are grid-adjacency on that "
        f"sample ({connectivity}-connected), NOT Euclidean k-NN and NOT the SPOC "
        "trajectory. SPOC actions use agent_move_m / agent_rotation_deg and may "
        "fail, detour, or backtrack; see navigation-*.csv. Door openness over the "
        "rollout is in doors-*.csv / passage_state-*.csv; object moves that may "
        "change navigability later are in displacement_events-*.csv."
    )
    if extra_notes:
        notes = notes + " " + extra_notes

    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "snapshot": snapshot,
        "coordinate_frame": {
            "system": "ai2thor",
            "y_up": True,
            "horizontal_plane": "xz",
            "units": "meters",
            "rotation_y_deg_cw_from_pos_z": True,
        },
        "params": {
            "grid_size": round(gs, 6),
            "agent_move_m": float(agent_move_m),
            "agent_rotation_deg": float(agent_rotation_deg),
            "edge_connectivity": connectivity,
            "edge_rule": (
                f"Connect reachable cells whose xz separation is at most one "
                f"{connectivity}-neighbor step on the GetReachablePositions grid "
                f"(max_step≈{max_step:.4f} m). Do not infer edges from trajectory."
            ),
            "thor_action": "GetReachablePositions",
            "snap_to_grid": False,
        },
        "door_states_at_snapshot": door_snapshot,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "notes": notes,
    }
