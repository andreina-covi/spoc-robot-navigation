"""Load configs/collection.yaml once. All collection tunables come from there."""
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml

from spoc_constants import ABS_PATH_OF_SPOC_DIR

DEFAULT_PATH = os.path.join(ABS_PATH_OF_SPOC_DIR, "configs", "collection.yaml")


@lru_cache(maxsize=None)
def load_collection_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the YAML. ``SPOC_COLLECTION_CONFIG`` overrides the default path."""
    path = path or os.environ.get("SPOC_COLLECTION_CONFIG") or DEFAULT_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def capped_horizon(max_steps: Optional[int], cap: Optional[int]) -> Optional[int]:
    """Clamp a task horizon to ``episode.max_policy_steps``.

    Non-positive ``max_steps`` means unset (online eval patches it later); a
    null or non-positive ``cap`` keeps the task horizon.
    """
    if cap is None or cap <= 0:
        return max_steps
    if max_steps is None or max_steps <= 0:
        return int(cap)
    return min(int(max_steps), int(cap))
