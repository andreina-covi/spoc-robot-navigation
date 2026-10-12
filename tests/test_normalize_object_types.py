"""Offline object-type normalization. No simulator."""

import csv
import json
from pathlib import Path

from scripts.normalize_object_types import (
    apply_mapping,
    build_mapping,
    last_word_category,
    lemma_to_camel,
    proposed_category,
)


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


def test_find_episodes_walks_nested_folders(tmp_path):
    from scripts.normalize_object_types import find_episodes

    nested = tmp_path / "files" / "run" / "annotations"
    nested.mkdir(parents=True)
    top = tmp_path / "run" / "annotations"
    top.mkdir(parents=True)
    found = find_episodes(tmp_path)
    assert found == [nested.parent, top.parent]


def test_camel_and_category_rules():
    assert lemma_to_camel("easy_chair") == "EasyChair"
    assert last_word_category("DiningTable") == "Table"
    assert last_word_category("FloorLamp") == "Lamp"
    assert proposed_category("ArmChair") == ("Chair", "override")
    assert proposed_category("CounterTop") == ("CounterTop", "override")
    assert proposed_category("Sofa") == ("Sofa", "last_word")


def test_build_keeps_manual_edits_and_apply_is_idempotent(tmp_path, monkeypatch):
    scene = "house_000007"
    episode = _episode(
        tmp_path,
        scene,
        [
            {"obj-type": "DiningTable", "obj-id": "DiningTable|1|0", "obj-color": "brown"},
            {"obj-type": "Undefined", "obj-id": "ObjaScooter|4|5", "obj-color": "white"},
            {"obj-type": "Undefined", "obj-id": "ObjaBooks|2|2", "obj-color": "red"},
            {"obj-type": "CounterTop", "obj-id": "CounterTop|1|0", "obj-color": "grey"},
        ],
    )
    state = episode / "annotations" / f"object_state-{scene}.csv"
    with state.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestep", "obj-id", "obj-type"])
        writer.writeheader()
        writer.writerow(
            {"timestep": "0", "obj-id": "ObjaBooks|2|2", "obj-type": "ObjaBooks"}
        )
    mapping = tmp_path / "type_mapping.csv"

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
        "scripts.normalize_object_types.get_objaverse_annotations",
        lambda: {
            "asset-scooter": {"most_specific_lemma": "kick_scooter"},
            "asset-books": {"most_specific_lemma": "comic_book"},
        },
    )

    build_mapping([episode], mapping, split_arg="val")
    saved = mapping.read_text()
    # Manual category edit must survive a second build.
    text = saved.replace("CounterTop,CounterTop,override,no", "CounterTop,Surface,override,yes")
    mapping.write_text(text)
    build_mapping([episode], mapping, split_arg="val")
    with mapping.open() as handle:
        rows = {row["obj_type"]: row for row in csv.DictReader(handle)}
    assert rows["CounterTop"]["obj_category"] == "Surface"
    assert rows["CounterTop"]["reviewed"] == "yes"
    assert rows["KickScooter"]["obj_category"] == "Scooter"
    assert rows["ComicBook"]["obj_category"] == "Book"
    assert "ObjaBooks" not in rows
    assert rows["DiningTable"]["obj_category"] == "Table"

    apply_mapping([episode], mapping, split_arg="val")
    objects = episode / "annotations" / f"objects-{scene}.csv"
    with objects.open() as handle:
        written = list(csv.DictReader(handle))
    by_id = {row["obj-id"]: row for row in written}
    assert by_id["DiningTable|1|0"]["obj-type"] == "DiningTable"
    assert by_id["DiningTable|1|0"]["obj-category"] == "Table"
    assert by_id["ObjaScooter|4|5"]["obj-type"] == "KickScooter"
    assert by_id["ObjaScooter|4|5"]["obj-category"] == "Scooter"
    assert by_id["ObjaBooks|2|2"]["obj-type"] == "ComicBook"
    assert by_id["ObjaBooks|2|2"]["obj-category"] == "Book"
    with state.open() as handle:
        state_rows = list(csv.DictReader(handle))
    assert state_rows[0]["obj-type"] == "ComicBook"
    assert state_rows[0]["obj-category"] == "Book"
    assert by_id["CounterTop|1|0"]["obj-category"] == "Surface"
    assert list(by_id["DiningTable|1|0"])[:2] == ["obj-type", "obj-category"]
    assert (episode / "annotations" / f"objects-{scene}.orig.csv").exists()

    apply_mapping([episode], mapping, split_arg="val")
    with objects.open() as handle:
        again = list(csv.DictReader(handle))
    assert again == written
