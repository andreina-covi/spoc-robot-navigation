"""Offline object-type normalization. No simulator."""

import csv
import json
from pathlib import Path

from scripts.normalize_object_types import (
    ObjectResolution,
    apply_episodes,
    confusable_pairs,
    find_episodes,
)
from utils.object_labels import describe_object, lemma_to_camel


def _episode(root: Path, scene: str, rows):
    ann = root / "episode" / "annotations"
    ann.mkdir(parents=True)
    (ann / f"episode_meta-{scene}.json").write_text(
        json.dumps({"scene_id": scene, "split": "val"})
    )
    with (ann / f"objects-{scene}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["obj-type", "obj-id", "obj-color"])
        writer.writeheader()
        writer.writerows(rows)
    return root / "episode"


def test_aabb_bounds_from_corners():
    from geometry import aabb_bounds

    low, high = aabb_bounds(
        {"cornerPoints": [[0, 1, 2], [4, 0, 3], [1, 5, 0]]}
    )
    assert low == (0.0, 0.0, 0.0)
    assert high == (4.0, 5.0, 3.0)


def test_find_episodes_walks_nested_folders(tmp_path):
    nested = tmp_path / "files" / "run" / "annotations"
    nested.mkdir(parents=True)
    top = tmp_path / "run" / "annotations"
    top.mkdir(parents=True)
    found = find_episodes(tmp_path)
    assert found == [nested.parent, top.parent]


def test_describe_object_keeps_numeric_id_meaning_in_the_type(monkeypatch):
    monkeypatch.setattr(
        "utils.object_labels.get_objaverse_annotations",
        lambda: {"asset-tv": {"most_specific_lemma": "television", "synset": "television_receiver.n.01"}},
    )
    thor = describe_object("6|5", "Television", None)
    assert thor.obj_type == "Television"
    assert thor.synset == "television_receiver.n.01"
    assert thor.source == "thor_type"
    named = describe_object("ObjaScooter|4|5", "Undefined", "asset-tv")
    assert named.obj_type == "Television"
    assert named.synset == "television_receiver.n.01"
    assert named.source == "objaverse_annotation"


def test_lemma_to_camel():
    assert lemma_to_camel("easy_chair") == "EasyChair"
    assert lemma_to_camel("scooter") == "Scooter"


def test_apply_writes_synset_and_is_idempotent(tmp_path, monkeypatch):
    scene = "house_000007"
    episode = _episode(
        tmp_path,
        scene,
        [
            {"obj-type": "DiningTable", "obj-id": "DiningTable|1|0", "obj-color": "brown"},
            {"obj-type": "Undefined", "obj-id": "ObjaScooter|4|5", "obj-color": "white"},
            {"obj-type": "Undefined", "obj-id": "ObjaBooks|2|2", "obj-color": "red"},
            {"obj-type": "CounterTop", "obj-id": "CounterTop|1|0", "obj-color": "grey"},
            {"obj-type": "Undefined", "obj-id": "ObjaWidget|9|1", "obj-color": "blue"},
        ],
    )
    state = episode / "annotations" / f"object_state-{scene}.csv"
    with state.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestep", "obj-id", "obj-type"])
        writer.writeheader()
        writer.writerow(
            {"timestep": "0", "obj-id": "ObjaBooks|2|2", "obj-type": "ObjaBooks"}
        )

    def asset_map(self, split, house_index):
        assert split == "val"
        assert house_index == 7
        return {
            "ObjaScooter|4|5": "asset-scooter",
            "ObjaBooks|2|2": "asset-books",
        }

    monkeypatch.setattr(
        "scripts.normalize_object_types.HouseIndex.asset_map", asset_map
    )
    monkeypatch.setattr(
        "utils.object_labels.get_objaverse_annotations",
        lambda: {
            "asset-scooter": {
                "most_specific_lemma": "kick_scooter",
                "synset": "scooter.n.01",
            },
            "asset-books": {
                "most_specific_lemma": "comic_book",
                "synset": "comic_book.n.01",
            },
        },
    )
    monkeypatch.setattr(
        "utils.object_labels.first_noun_synset",
        lambda name: "widget.n.01" if name == "Widget" else "",
    )
    monkeypatch.setattr(
        "scripts.normalize_object_types.confusable_pairs", lambda resolved: []
    )

    apply_episodes([episode], split_arg="val")
    objects = episode / "annotations" / f"objects-{scene}.csv"
    with objects.open() as handle:
        written = list(csv.DictReader(handle))
    by_id = {row["obj-id"]: row for row in written}
    assert by_id["DiningTable|1|0"]["obj-type"] == "DiningTable"
    assert by_id["DiningTable|1|0"]["obj-synset"] == "dining_table.n.01"
    assert by_id["ObjaScooter|4|5"]["obj-type"] == "KickScooter"
    assert by_id["ObjaScooter|4|5"]["obj-synset"] == "scooter.n.01"
    assert by_id["ObjaBooks|2|2"]["obj-type"] == "ComicBook"
    assert by_id["ObjaBooks|2|2"]["obj-synset"] == "comic_book.n.01"
    assert by_id["CounterTop|1|0"]["obj-synset"] == "countertop.n.01"
    assert by_id["ObjaWidget|9|1"]["obj-type"] == "Widget"
    assert by_id["ObjaWidget|9|1"]["obj-synset"] == "widget.n.01"
    assert "obj-category" not in by_id["DiningTable|1|0"]
    assert list(by_id["DiningTable|1|0"])[:2] == ["obj-type", "obj-synset"]
    with state.open() as handle:
        state_rows = list(csv.DictReader(handle))
    assert state_rows[0]["obj-type"] == "ComicBook"
    assert state_rows[0]["obj-synset"] == "comic_book.n.01"
    report = list(
        csv.DictReader((episode / "annotations" / "type_normalization_report.csv").open())
    )
    sources = {row["obj-id"]: row["source"] for row in report}
    assert sources["DiningTable|1|0"] == "thor_type"
    assert sources["ObjaScooter|4|5"] == "objaverse_annotation"
    assert sources["ObjaWidget|9|1"] == "id_name"
    assert (episode / "annotations" / f"objects-{scene}.orig.csv").exists()

    apply_episodes([episode], split_arg="val")
    with objects.open() as handle:
        again = list(csv.DictReader(handle))
    assert again == written


def test_confusable_pairs(monkeypatch):
    parents = {
        "cup.n.01": ["vessel.n.03"],
        "mug.n.04": ["cup.n.01"],
        "refrigerator.n.01": ["white_goods.n.01"],
        "dining_table.n.01": ["table.n.02"],
        "coffee_table.n.01": ["table.n.02"],
    }

    class _Synset:
        def __init__(self, name):
            self._name = name

        def hypernyms(self):
            return [_Synset(parent) for parent in parents.get(self._name, [])]

        def name(self):
            return self._name

    def fake_hypernym(synset, possible_hypernym):
        return possible_hypernym in parents.get(synset, [])

    monkeypatch.setattr("scripts.normalize_object_types.wn.synset", _Synset)
    monkeypatch.setattr("scripts.normalize_object_types.is_hypernym_of", fake_hypernym)

    items = [
        ObjectResolution("cup", "Cup", "Cup", "cup.n.01", "thor_type"),
        ObjectResolution("mug", "Mug", "Mug", "mug.n.04", "thor_type"),
        ObjectResolution("fridge", "Fridge", "Fridge", "refrigerator.n.01", "thor_type"),
        ObjectResolution(
            "obja-fridge", "Undefined", "Refrigerator", "refrigerator.n.01", "objaverse_annotation"
        ),
        ObjectResolution("dining", "DiningTable", "DiningTable", "dining_table.n.01", "thor_type"),
        ObjectResolution("coffee", "CoffeeTable", "CoffeeTable", "coffee_table.n.01", "thor_type"),
    ]
    reasons = {
        (left.obj_id, right.obj_id): reason for left, right, reason in confusable_pairs(items)
    }
    assert reasons[("fridge", "obja-fridge")] == "same_synset"
    assert reasons[("cup", "mug")] == "hypernym"
    assert reasons[("dining", "coffee")] == "shared_direct_hypernym"
    assert ("cup", "fridge") not in reasons
    assert ("mug", "dining") not in reasons


def test_motion_uses_exported_action_names(tmp_path, monkeypatch):
    monkeypatch.setattr("collector.OBJAVERSE_NAVIGATION_PATH", str(tmp_path))
    from collector import Collector

    collector = Collector(scene_name="house_000001", max_steps=2)
    collector._note_motion("move_ahead")
    collector._note_motion("rotate_left_small")
    collector.nav_graphs["episode_start"] = {
        "params": {
            "grid_size": 0.15,
            "agent_move_m": 0.2,
            "agent_rotation_deg": 45.0,
        }
    }
    report = collector._motion_check()
    assert report["move_m_used"] == [0.2]
    assert report["rotation_deg_used"] == [9.0]
    assert collector._rotation_degrees("rotate_left") == 45.0
    assert collector._rotation_degrees("move_ahead") is None
    assert report["matches"] is False
    assert "rotation angles used" in report["report"]
