from utils.placement_anchors import (
    floor_anchor_points,
    receptacle_salience,
    salient_landmarks,
    salient_receptacles,
)


def _box(x, y, z):
    return {"axisAlignedBoundingBox": {"size": {"x": x, "y": y, "z": z}}}


def test_salience_is_box_volume():
    assert receptacle_salience(_box(2, 3, 4)) == 24


def test_top_receptacles_skip_openable_and_structural():
    objects = [
        {"objectId": "Counter|1", "objectType": "CounterTop", "receptacle": True, "room_id": "room|1", **_box(2, 1, 2)},
        {"objectId": "Sofa|1", "objectType": "Sofa", "receptacle": True, "room_id": "room|1", **_box(3, 1, 1)},
        {"objectId": "Fridge|1", "objectType": "Fridge", "receptacle": True, "openable": True, "room_id": "room|1", **_box(9, 9, 9)},
        {"objectId": "Floor|1", "objectType": "Floor", "receptacle": True, "room_id": "room|1", **_box(20, 1, 20)},
        {"objectId": "Cup|1", "objectType": "Cup", "receptacle": True, "pickupable": True, "room_id": "room|1", **_box(1, 1, 1)},
        {"objectId": "Bed|2", "objectType": "Bed", "receptacle": True, "room_id": "room|2", **_box(4, 1, 2)},
    ]
    chosen = salient_receptacles(objects, "room|1", limit=5)
    assert [obj["objectId"] for obj in chosen] == ["Counter|1", "Sofa|1"]


def test_floor_anchor_is_the_closest_spawn_inside_the_radius():
    landmarks = [{"landmark_id": "Sofa|1", "position": {"x": 0, "y": 0, "z": 0}}]
    spawns = [
        {"x": 2.0, "y": 0.1, "z": 0},
        {"x": 0.4, "y": 0.1, "z": 0.1},
        {"x": 0.8, "y": 0.1, "z": 0},
    ]
    anchors = floor_anchor_points(landmarks, spawns, radius_m=1.2)
    assert len(anchors) == 1
    assert anchors[0][0] == "Sofa|1"
    assert anchors[0][1]["x"] == 0.4


def test_landmarks_are_non_pickupable_ranked_by_volume():
    objects = [
        {"objectId": "ObjaTable|1", "objectType": "ObjaTable", "room_id": "room|1", **_box(2, 1, 2)},
        {"objectId": "Chair|1", "objectType": "Chair", "room_id": "room|1", **_box(1, 1, 1)},
        {"objectId": "Cup|1", "objectType": "Cup", "pickupable": True, "room_id": "room|1", **_box(3, 3, 3)},
        {"objectId": "Floor|1", "objectType": "Floor", "room_id": "room|1", **_box(9, 1, 9)},
        {"objectId": "Sofa|2", "objectType": "Sofa", "room_id": "room|2", **_box(4, 1, 2)},
    ]
    chosen = salient_landmarks(objects, "room|1")
    assert [obj["objectId"] for obj in chosen] == ["ObjaTable|1", "Chair|1"]


def test_candidates_name_includes_split_and_radius():
    from scripts.placement_trials import candidates_name

    assert (
        candidates_name(1030, "val", 1.2)
        == "placement_candidates-house_001030-val-radius_1.2.csv"
    )


def test_houses_from_samples_keeps_unique_indexes():
    from scripts.placement_trials import houses_from_samples

    samples = [
        {"house_id": "007514", "house_split": "val"},
        {"house_index": 6068, "house_split": "val"},
        {"house_id": "7514", "house_split": "val"},
    ]
    assert houses_from_samples(samples) == {"val": [7514, 6068]}
