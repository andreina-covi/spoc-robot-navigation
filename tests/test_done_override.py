"""configs/collection.yaml episode.continue_until_all_rooms_visited."""
from types import SimpleNamespace

import pytest

from utils.collection_config import load_collection_config
from utils.type_utils import THORActions


def test_config_has_override_settings():
    episode = load_collection_config()["episode"]
    assert isinstance(episode["continue_until_all_rooms_visited"], bool)
    assert episode["max_done_overrides"] >= 0
    assert hasattr(THORActions, episode["done_override_action"])


def _task(visited, rooms=("room|2", "room|3"), enabled=True, cap=2):
    try:
        from tasks.room_visit_task import RoomVisitTask
    except Exception as exc:  # assets or simulator deps missing
        pytest.skip(f"RoomVisitTask not importable: {exc}")
    task = RoomVisitTask.__new__(RoomVisitTask)
    task._collection_cfg = {
        "episode": {
            "continue_until_all_rooms_visited": enabled,
            "max_done_overrides": cap,
            "done_override_action": "rotate_right",
        }
    }
    task.room_poly_map = {r: None for r in rooms}
    task.collector = SimpleNamespace(visited_rooms=set(visited), done_overrides=0)
    task._done_overrides = 0
    return task


def test_done_becomes_turn_while_a_room_is_unvisited():
    task = _task({"room|2"})
    assert task._override_done(THORActions.done) == THORActions.rotate_right
    assert task.collector.done_overrides == 1


def test_done_accepted_when_every_room_visited():
    task = _task({"room|2", "room|3"})
    assert task._override_done(THORActions.done) == THORActions.done


def test_done_accepted_after_cap():
    task = _task({"room|2"}, cap=2)
    task._override_done(THORActions.done)
    task._override_done(THORActions.done)
    assert task._override_done(THORActions.done) == THORActions.done
    assert task.collector.done_overrides == 2


def test_disabled_and_other_actions_untouched():
    assert _task({"room|2"}, enabled=False)._override_done(THORActions.done) == THORActions.done
    assert _task({"room|2"})._override_done(THORActions.move_ahead) == THORActions.move_ahead
