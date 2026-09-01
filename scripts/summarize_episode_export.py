#!/usr/bin/env python3
"""Summarize one exported RoomVisit / invisible_displacement episode.

Usage:
  python scripts/summarize_episode_export.py --run_dir "$OBJAVERSE_NAVIGATION_PATH/<timestamp>"
  python scripts/summarize_episode_export.py --run_dir ... --scene house_000012
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _find_scene(annotations_dir: str, scene: Optional[str]) -> str:
    if scene:
        return scene
    metas = [f for f in os.listdir(annotations_dir) if f.startswith("episode_meta-") and f.endswith(".json")]
    if not metas:
        raise FileNotFoundError(f"No episode_meta-*.json in {annotations_dir}")
    # episode_meta-house_XXXXXX.json
    name = metas[0][len("episode_meta-") : -len(".json")]
    return name


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _agent_rows(nav: pd.DataFrame) -> pd.DataFrame:
    """One row per timestep (agent pose / action), ignoring FOV object rows."""
    if nav.empty:
        return nav
    # Prefer first row per timestep (agent fields are repeated on object rows)
    cols = [c for c in ("timestep", "ag-action", "action_success", "ag-pos-x", "ag-pos-z") if c in nav.columns]
    return nav[cols].drop_duplicates(subset=["timestep"], keep="first").sort_values("timestep")


def summarize(run_dir: str, scene: Optional[str] = None) -> Dict[str, Any]:
    annotations = os.path.join(run_dir, "annotations")
    if not os.path.isdir(annotations):
        raise FileNotFoundError(f"Missing annotations/ under {run_dir}")

    scene = _find_scene(annotations, scene)
    meta = _load_json(os.path.join(annotations, f"episode_meta-{scene}.json")) or {}
    nav_path = os.path.join(annotations, f"navigation-{scene}.csv")
    nav = pd.read_csv(nav_path) if os.path.exists(nav_path) else pd.DataFrame()
    agent = _agent_rows(nav)

    n_steps = int(agent["timestep"].nunique()) if len(agent) and "timestep" in agent.columns else 0
    if len(agent) and "action_success" in agent.columns:
        # action_success may be bool / 0-1 / -1 (True==1 in pandas — count once)
        succ = pd.to_numeric(agent["action_success"], errors="coerce")
        n_success = int((succ == 1).sum())
        n_fail = int((succ == 0).sum())
    else:
        n_success = n_fail = 0

    unique_pos = 0
    if len(agent) and "ag-pos-x" in agent.columns and "ag-pos-z" in agent.columns:
        rounded = list(
            zip(
                agent["ag-pos-x"].round(3),
                agent["ag-pos-z"].round(3),
            )
        )
        unique_pos = len(set(rounded))

    nav_graph = _load_json(os.path.join(annotations, f"nav_graph-{scene}.json")) or {}
    snaps = nav_graph.get("snapshots") or {}
    start = snaps.get("episode_start") or {}
    end = snaps.get("episode_end") or {}

    # Trajectory vs graph consistency: agent xz should lie near some reachable node
    nearest_ok = None
    if start.get("nodes") and unique_pos:
        nodes = start["nodes"]
        nx = [n["x"] for n in nodes]
        nz = [n["z"] for n in nodes]
        max_d = None
        for x, z in set(rounded):
            d = min((x - a) ** 2 + (z - b) ** 2 for a, b in zip(nx, nz)) ** 0.5
            max_d = d if max_d is None else max(max_d, d)
        nearest_ok = {
            "max_agent_to_nearest_start_node_m": round(max_d, 4) if max_d is not None else None,
            "note": "Large values can mean agent left the start reachability set (doors/objects) or pose noise.",
        }

    out = {
        "run_dir": run_dir,
        "scene_id": scene,
        "episode_id": meta.get("episode_id"),
        "num_timesteps": n_steps,
        "meta_num_timesteps": meta.get("num_timesteps"),
        "successful_actions": n_success,
        "failed_actions": n_fail,
        "unique_xz_positions_visited": unique_pos,
        "nav_graph_start_nodes": start.get("num_nodes"),
        "nav_graph_start_edges": start.get("num_edges"),
        "nav_graph_end_nodes": end.get("num_nodes"),
        "nav_graph_end_edges": end.get("num_edges"),
        "reachability_params": start.get("params") or end.get("params"),
        "trajectory_vs_reachability": nearest_ok,
        "save_reason": meta.get("save_reason"),
        "num_displacements": meta.get("num_displacements"),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Episode root (.../<timestamp>/)")
    p.add_argument("--scene", default=None, help="e.g. house_000012 (default: detect)")
    args = p.parse_args(argv)
    summary = summarize(args.run_dir, args.scene)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
