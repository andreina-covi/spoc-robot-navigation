"""Which houses a collection folder already holds, so a chunked export can resume."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def finished_episodes(nav_root) -> Dict[Tuple[str, int], Path]:
    """``(split, house_index)`` -> annotations folder of each finished episode.

    An episode is finished once the collector wrote ``episode_meta-*.json``
    (``Collector.save_data``, the last step). A folder without it is partial.
    """
    found: Dict[Tuple[str, int], Path] = {}
    root = Path(nav_root)
    if not root.is_dir():
        return found
    for meta_path in sorted(root.glob("*/annotations/episode_meta-*.json")):
        try:
            meta = json.loads(meta_path.read_text())
            key = (str(meta["split"]), int(meta["house_index"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        found.setdefault(key, meta_path.parent)
    return found


def has_placement_trials(annotations_dir) -> bool:
    return any(Path(annotations_dir).glob("placement_candidates-*.csv"))


def set_aside_partial_episodes(nav_root, house_index: int) -> List[Path]:
    """Rename this house's unfinished folders to ``.partial-<name>`` before a new walk.

    Leaves the data in place, but a dot name keeps it out of cm-benchmark stage 1
    (``list_collection_episodes`` skips dot folders) and out of the house's name.
    """
    root = Path(nav_root)
    name_re = re.compile(rf"house_{int(house_index):06d}(_\d+)?")
    moved: List[Path] = []
    if not root.is_dir():
        return moved
    for folder in sorted(root.glob(f"house_{int(house_index):06d}*")):
        if not folder.is_dir() or not name_re.fullmatch(folder.name):
            continue
        if any((folder / "annotations").glob("episode_meta-*.json")):
            continue
        target = root / f".partial-{folder.name}"
        n = 2
        while target.exists():
            target = root / f".partial-{folder.name}-{n}"
            n += 1
        folder.rename(target)
        moved.append(target)
    return moved
