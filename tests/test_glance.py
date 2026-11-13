import math

import pytest

from utils.collection_config import load_collection_config
from utils.glance import (
    GlancePlanner,
    GlanceSettings,
    glance_candidates,
    horizontal_fov_deg,
    is_hidden_in_closed_receptacle,
)

W, H, VFOV, ROT = 396, 224, 59.0, 45.0


def settings(**over):
    cfg = dict(load_collection_config()["glance"])
    cfg.update(over)
    return GlanceSettings.from_config(cfg)


def obj(oid, x, z, size=0.4, room="r1"):
    return {"id": oid, "x": x, "z": z, "width": size, "height": size, "room": room}


def planner(objects, **over):
    return GlancePlanner(objects, settings(**over), W, H, VFOV, ROT)


def test_config_has_every_key():
    cfg = load_collection_config()
    assert cfg["episode"]["timeout_s"] > 0
    GlanceSettings.from_config(cfg["glance"])  # raises on a missing key


def test_hfov_about_90():
    assert 85 < horizontal_fov_deg(W, H, VFOV) < 95


def test_right_object_out_of_view_gets_one_right_turn():
    # agent at origin facing +z (yaw 0); +x is to the right
    t = planner([obj("box", 1.0, 0.8)]).pick(0, 0, 0, "r1", step=0)  # ~51°
    assert t.object_id == "box" and t.turns == 1


def test_left_and_two_turns():
    t = planner([obj("box", -1.0, -0.4)]).pick(0, 0, 0, "r1", step=0)  # ~-112°
    assert t.turns == -2


def test_in_view_behind_other_room_and_small_are_skipped():
    p = planner(
        [
            obj("ahead", 0.0, 1.0),  # already in view
            obj("behind", 0.0, -1.0),  # beyond max_turns
            obj("other_room", 1.0, 0.0, room="r2"),
            obj("cup", 1.5, 0.0, size=0.1),  # too small at 1.5 m
        ]
    )
    assert p.pick(0, 0, 0, "r1", step=0) is None


def test_small_object_triggers_when_close():
    p = planner([obj("cup", 0.5, 0.0, size=0.1)])
    assert p.pick(0, 0, 0, "r1", step=0) is not None


def test_limits_once_per_object_spacing_and_episode_cap():
    p = planner([obj("a", 1.0, 0.0), obj("b", 1.0, 0.2)], max_per_episode=2, min_steps_between=5)
    t = p.pick(0, 0, 0, "r1", step=0)
    p.mark_done(t, 0)
    assert p.pick(0, 0, 0, "r1", step=3) is None  # too soon
    t2 = p.pick(0, 0, 0, "r1", step=5)
    assert t2.object_id != t.object_id
    p.mark_done(t2, 5)
    assert p.pick(0, 0, 0, "r1", step=20) is None  # both glanced, cap reached


def test_closed_receptacle_hides_object():
    by_id = {"fridge": {"openable": True, "isOpen": False}, "table": {}}
    assert is_hidden_in_closed_receptacle({"parentReceptacles": ["fridge"]}, by_id)
    assert not is_hidden_in_closed_receptacle({"parentReceptacles": ["table"]}, by_id)


@pytest.mark.parametrize("bearing", [50, 60, 67, 70, 90, 112])
def test_residual_after_turns_is_inside_view(bearing):
    p = planner([])
    n = p.turns_for(bearing)
    assert abs(bearing - n * ROT) < p.half_hfov


def thor_obj(oid, **over):
    base = {
        "objectId": oid,
        "objectType": oid.split("|")[0],
        "pickupable": False,
        "moveable": False,
        "mass": 1.0,
        "receptacleObjectIds": [],
        "parentReceptacles": [],
        "position": {"x": 0.0, "y": 0.5, "z": 0.0},
        "axisAlignedBoundingBox": {
            "center": {"x": 0.0, "y": 0.5, "z": 0.0},
            "size": {"x": 0.5, "y": 1.0, "z": 0.5},
        },
    }
    base.update(over)
    return base


def test_glance_candidates_are_the_placement_trial_objects():
    objects = [
        thor_obj("Mug|1", pickupable=True),
        thor_obj("Chair|1", moveable=True, mass=10.0),
        thor_obj("Sofa|1", moveable=True, mass=104.0),  # too heavy
        thor_obj("Desk|1", moveable=True, mass=20.0, receptacleObjectIds=["Mug|1"]),  # loaded
        thor_obj("Fridge|1"),  # neither pickupable nor moveable
        thor_obj("Drawer|1", openable=True, isOpen=False),
        thor_obj("Pen|1", pickupable=True, parentReceptacles=["Drawer|1"]),  # hidden
        thor_obj("Plate|1", pickupable=True, axisAlignedBoundingBox={}),  # no box
    ]
    ids = lambda include: [o["objectId"] for o in glance_candidates(objects, include, 30.0)]
    assert ids(True) == ["Mug|1", "Chair|1"]
    assert ids(False) == ["Mug|1"]
