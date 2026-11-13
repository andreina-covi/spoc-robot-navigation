"""configs/collection.yaml episode.max_policy_steps clamps the RoomVisit horizon."""
import pytest

from utils.collection_config import capped_horizon, load_collection_config


def test_config_sets_a_cap():
    assert load_collection_config()["episode"]["max_policy_steps"] > 0


@pytest.mark.parametrize(
    "max_steps,cap,expected",
    [(1000, 600, 600), (300, 600, 300), (-1, 600, 600), (None, 600, 600), (1000, None, 1000)],
)
def test_cap_clamps_horizon(max_steps, cap, expected):
    assert capped_horizon(max_steps, cap) == expected


def test_room_visit_task_clamps_its_horizon():
    """The real task method, when the THOR/objaverse imports are available."""
    from types import SimpleNamespace

    try:
        from tasks.room_visit_task import RoomVisitTask
    except Exception as exc:  # assets or simulator deps missing
        pytest.skip(f"RoomVisitTask not importable: {exc}")
    task = RoomVisitTask.__new__(RoomVisitTask)
    task.max_steps = 1000
    task._collection_cfg = {"episode": {"max_policy_steps": 600}}
    task._glance_settings = None
    task.collector = SimpleNamespace(max_steps=None, fov_stride=None)
    task.controller = SimpleNamespace()
    task._sync_collector_horizon()
    assert task.max_steps == 600
    assert task.collector.max_steps == 600
