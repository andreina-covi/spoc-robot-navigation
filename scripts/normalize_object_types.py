#!/usr/bin/env python3
"""Fill descriptive object types and WordNet synsets on exported episodes.

Reads annotation files already written by the collector. Does not start
AI2-THOR or an agent.

--apply PATH
    Write obj-type and obj-synset. THOR types use
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET. Objaverse assets use
    annotation["synset"]. If the asset cannot be resolved, the name after
    "Obja" in the obj-id is looked up as the first wordnet2022 noun synset
    and the report source is id_name.

The episode metadata often has a house index (scene_id house_001030) but no
split. This script does not guess the split. Pass --split val (or train/test)
when you know which OBJAVERSE_HOUSES_DIR/<split>.jsonl.gz was used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nltk.corpus import wordnet2022 as wn  # noqa: E402

from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR  # noqa: E402
from utils.object_labels import (  # noqa: E402
    describe_object,
    is_id_placeholder,
    is_undefined,
)
from utils.synset_utils import is_hypernym_of  # noqa: E402

REPORT_NAME = "type_normalization_report.csv"
REPORT_FIELDS = [
    "obj-id",
    "original obj-type",
    "new obj-type",
    "obj-synset",
    "source",
]

TYPE_COLUMNS = ("obj-type", "objectType", "obj_type")
ID_COLUMNS = ("obj-id", "objectId", "obj_id", "landmark_id")
SYNSET_COLUMN = "obj-synset"
# Dropped when rewriting files that an earlier run marked with a category.
_RETIRED_COLUMNS = ("obj-category",)
SPLIT_KEYS = ("split", "house_split", "dataset_split")
HOUSE_INDEX_KEYS = ("house_index", "houseIndex")
HOUSE_SPLITS = ("train", "val", "test")

@dataclass
class ObjectResolution:
    obj_id: str
    original_type: str
    descriptive: str
    synset: str
    source: str  # thor_type | objaverse_annotation | id_name


def parse_house_index(scene_id: str) -> Optional[int]:
    if not scene_id:
        return None
    match = re.search(r"(\d+)$", str(scene_id))
    if not match:
        return None
    return int(match.group(1))


def find_episodes(path: Path) -> List[Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir() and path.name == "annotations":
        return [path.parent]
    if (path / "annotations").is_dir():
        return [path]
    episodes = [
        annotations.parent
        for annotations in sorted(path.rglob("annotations"))
        if annotations.is_dir()
    ]
    if not episodes:
        raise FileNotFoundError(
            f"No episode folders with annotations/ under {path}"
        )
    return episodes


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def episode_house_identity(annotations: Path) -> Tuple[Optional[int], Optional[str], str]:
    """Return (house_index, split or None, note).

    Split is taken only from episode JSON. It is never inferred.
    """
    notes: List[str] = []
    house_index: Optional[int] = None
    split: Optional[str] = None
    scene_id = ""

    json_files = sorted(annotations.glob("*.json"))
    for path in json_files:
        if ".orig." in path.name or path.name == REPORT_NAME:
            continue
        try:
            data = _load_json(path)
        except json.JSONDecodeError:
            continue
        found_index, found_split, found_scene = _scan_identity(data)
        if house_index is None and found_index is not None:
            house_index = found_index
        if split is None and found_split:
            split = found_split
        if not scene_id and found_scene:
            scene_id = found_scene

    if house_index is None and scene_id:
        house_index = parse_house_index(scene_id)
    if house_index is None:
        notes.append("house index not in episode metadata")
    if split is None:
        notes.append("split not in episode metadata")
    return house_index, split, "; ".join(notes)


def _scan_identity(node: Any) -> Tuple[Optional[int], Optional[str], str]:
    house_index = None
    split = None
    scene_id = ""
    if isinstance(node, dict):
        for key in HOUSE_INDEX_KEYS:
            if key in node and house_index is None:
                try:
                    house_index = int(node[key])
                except (TypeError, ValueError):
                    pass
        for key in SPLIT_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value in HOUSE_SPLITS and split is None:
                split = value
        scene = node.get("scene_id") or node.get("sceneId")
        if isinstance(scene, str):
            scene_id = scene
            if house_index is None:
                house_index = parse_house_index(scene)
        for value in node.values():
            child_index, child_split, child_scene = _scan_identity(value)
            if house_index is None:
                house_index = child_index
            if split is None:
                split = child_split
            if not scene_id:
                scene_id = child_scene
    elif isinstance(node, list):
        for value in node:
            child_index, child_split, child_scene = _scan_identity(value)
            if house_index is None:
                house_index = child_index
            if split is None:
                split = child_split
            if not scene_id:
                scene_id = child_scene
    return house_index, split, scene_id


class HouseIndex:
    """obj-id -> assetId for houses addressed by split and line index."""

    def __init__(self, houses_dir: Optional[str]):
        self.houses_dir = Path(houses_dir) if houses_dir else None
        self._cache: Dict[Tuple[str, int], Dict[str, str]] = {}

    def asset_map(self, split: str, house_index: int) -> Dict[str, str]:
        key = (split, house_index)
        if key in self._cache:
            return self._cache[key]
        mapping = self._load(split, house_index)
        self._cache[key] = mapping
        return mapping

    def _load(self, split: str, house_index: int) -> Dict[str, str]:
        if self.houses_dir is None:
            raise FileNotFoundError("OBJAVERSE_HOUSES_DIR is not set")
        path = self.houses_dir / f"{split}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        import gzip

        with gzip.open(path, "rt") as handle:
            for line_index, line in enumerate(handle):
                if line_index == house_index:
                    house = json.loads(line)
                    break
            else:
                raise IndexError(
                    f"{path} has no house at line index {house_index}"
                )
        mapping: Dict[str, str] = {}

        def walk(nodes: Iterable[dict]) -> None:
            for node in nodes or []:
                if not isinstance(node, dict):
                    continue
                obj_id = node.get("id")
                asset_id = node.get("assetId")
                if obj_id and asset_id:
                    mapping[str(obj_id)] = str(asset_id)
                children = node.get("children") or []
                if children:
                    walk(children)

        walk(house.get("objects") or [])
        return mapping


def orig_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.orig{path.suffix}")


def source_path(path: Path) -> Path:
    """Read the untouched export when a previous run already backed it up."""
    backup = orig_path(path)
    if backup.exists():
        return backup
    return path


def backup_once(path: Path) -> None:
    backup = orig_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)


def annotation_files(annotations: Path) -> List[Path]:
    files = []
    for path in sorted(annotations.iterdir()):
        if not path.is_file():
            continue
        if ".orig." in path.name or path.name.endswith(".orig"):
            continue
        if path.name == REPORT_NAME:
            continue
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        files.append(path)
    return files


def _type_column(fieldnames: Sequence[str]) -> Optional[str]:
    for name in TYPE_COLUMNS:
        if name in fieldnames:
            return name
    return None


def _id_column(fieldnames: Sequence[str]) -> Optional[str]:
    for name in ID_COLUMNS:
        if name in fieldnames:
            return name
    return None


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with source_path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def collect_pairs_from_csv(path: Path) -> List[Tuple[str, str]]:
    fieldnames, rows = read_csv_rows(path)
    type_col = _type_column(fieldnames)
    id_col = _id_column(fieldnames)
    if not type_col or not id_col:
        return []
    pairs = []
    for row in rows:
        obj_id = (row.get(id_col) or "").strip()
        if not obj_id:
            continue
        pairs.append((obj_id, row.get(type_col) or ""))
    return pairs


def collect_pairs_from_json(node: Any, pairs: List[Tuple[str, str]]) -> None:
    if isinstance(node, dict):
        type_key = next((key for key in TYPE_COLUMNS if key in node), None)
        id_key = next((key for key in ID_COLUMNS if key in node), None)
        if type_key and id_key and node.get(id_key):
            pairs.append((str(node[id_key]), str(node.get(type_key) or "")))
        for value in node.values():
            collect_pairs_from_json(value, pairs)
    elif isinstance(node, list):
        for value in node:
            collect_pairs_from_json(value, pairs)


def collect_object_pairs(annotations: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for path in annotation_files(annotations):
        if path.suffix.lower() == ".csv":
            pairs.extend(collect_pairs_from_csv(path))
        elif path.suffix.lower() == ".json":
            try:
                data = _load_json(source_path(path))
            except json.JSONDecodeError:
                continue
            collect_pairs_from_json(data, pairs)
    return pairs


def resolve_objects(
    pairs: Sequence[Tuple[str, str]],
    house_index: Optional[int],
    split: Optional[str],
    houses: HouseIndex,
    identity_note: str,
) -> List[ObjectResolution]:
    by_id: Dict[str, List[str]] = {}
    for obj_id, obj_type in pairs:
        by_id.setdefault(obj_id, []).append(obj_type)

    asset_map: Dict[str, str] = {}
    lookup_note = identity_note
    if house_index is not None and split:
        try:
            asset_map = houses.asset_map(split, house_index)
            lookup_note = ""
        except (FileNotFoundError, IndexError, json.JSONDecodeError) as exc:
            lookup_note = str(exc)

    resolved = []
    for obj_id in sorted(by_id):
        types = by_id[obj_id]
        real = [
            value
            for value in types
            if not is_undefined(value) and not is_id_placeholder(obj_id, value)
        ]
        asset_id = "" if lookup_note else asset_map.get(obj_id, "")
        if real:
            descriptive = str(real[0])
            label = describe_object(obj_id, descriptive, asset_id)
            source = "thor_type"
            original = descriptive
            synset = label.synset
        else:
            label = describe_object(obj_id, "Undefined", asset_id)
            descriptive = label.obj_type
            source = label.source
            synset = label.synset
            original = "Undefined" if any(is_undefined(value) for value in types) else (
                str(types[0]) if types else ""
            )
        resolved.append(
            ObjectResolution(
                obj_id=obj_id,
                original_type=original,
                descriptive=descriptive,
                synset=synset,
                source=source,
            )
        )
    return resolved


def resolve_episode(
    episode: Path,
    houses: HouseIndex,
    split_arg: Optional[str],
) -> Tuple[List[ObjectResolution], str]:
    ann = episode / "annotations"
    house_index, meta_split, note = episode_house_identity(ann)
    split = meta_split or (split_arg if split_arg in HOUSE_SPLITS else None)
    if meta_split is None and split_arg:
        note = ""
    elif meta_split is None and not split_arg:
        note = note or "split not in episode metadata"
    pairs = collect_object_pairs(ann)
    resolved = resolve_objects(
        pairs, house_index, split, houses, note if not split or house_index is None else ""
    )
    if not split or house_index is None:
        # resolve_objects already used id_name; keep the note for the log
        return resolved, note
    return resolved, ""


def apply_episodes(episodes: Sequence[Path], split_arg: Optional[str]) -> None:
    houses = HouseIndex(OBJAVERSE_HOUSES_DIR)
    for episode in episodes:
        resolved, note = resolve_episode(episode, houses, split_arg)
        if note:
            print(f"{episode.name}: {note}")
        write_episode(episode, resolved)
        print_confusable(episode, resolved)


def write_episode(episode: Path, resolved: Sequence[ObjectResolution]) -> None:
    by_id = {item.obj_id: item for item in resolved}
    annotations = episode / "annotations"
    changed: List[str] = []
    for path in annotation_files(annotations):
        if path.suffix.lower() == ".csv":
            fields = apply_csv(path, by_id)
        else:
            fields = apply_json(path, by_id)
        if fields:
            changed.append(f"{path.name}: {', '.join(fields)}")
    report_path = annotations / REPORT_NAME
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for item in resolved:
            writer.writerow(
                {
                    "obj-id": item.obj_id,
                    "original obj-type": item.original_type,
                    "new obj-type": item.descriptive,
                    "obj-synset": item.synset,
                    "source": item.source,
                }
            )
    print(f"{episode.name}")
    if changed:
        for line in changed:
            print(f"  updated {line}")
    else:
        print("  no type fields to update")
    verify_episode(annotations, by_id)


def apply_csv(path: Path, by_id: Dict[str, ObjectResolution]) -> List[str]:
    fieldnames, rows = read_csv_rows(path)
    type_col = _type_column(fieldnames)
    id_col = _id_column(fieldnames)
    if not type_col or not id_col:
        return []
    original_ids = [(row.get(id_col) or "") for row in rows]
    original_count = len(rows)
    out_fields = _insert_synset_column(fieldnames, type_col)
    for row in rows:
        obj_id = (row.get(id_col) or "").strip()
        item = by_id.get(obj_id)
        if item is None:
            row.setdefault(SYNSET_COLUMN, "")
            continue
        row[type_col] = item.descriptive
        row[SYNSET_COLUMN] = item.synset
    backup_once(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in out_fields})
    _assert_ids_unchanged(path, id_col, original_ids, original_count)
    return [type_col, SYNSET_COLUMN]


def apply_json(path: Path, by_id: Dict[str, ObjectResolution]) -> List[str]:
    data = _load_json(source_path(path))
    before_ids = _json_object_ids(data)
    fields: List[str] = []
    if not _json_has_type(data):
        return []
    _update_json(data, by_id, fields)
    if not fields:
        return []
    backup_once(path)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    after_ids = _json_object_ids(_load_json(path))
    if after_ids != before_ids:
        raise RuntimeError(f"{path.name}: object ids changed")
    return sorted(set(fields))


def _json_object_ids(node: Any, found: Optional[List[str]] = None) -> List[str]:
    if found is None:
        found = []
    if isinstance(node, dict):
        for key in ID_COLUMNS:
            if node.get(key):
                found.append(str(node[key]))
                break
        for value in node.values():
            _json_object_ids(value, found)
    elif isinstance(node, list):
        for value in node:
            _json_object_ids(value, found)
    return found


def _json_has_type(node: Any) -> bool:
    if isinstance(node, dict):
        if any(key in node for key in TYPE_COLUMNS):
            return True
        return any(_json_has_type(value) for value in node.values())
    if isinstance(node, list):
        return any(_json_has_type(value) for value in node)
    return False


def _update_json(node: Any, by_id: Dict[str, ObjectResolution], fields: List[str]) -> None:
    if isinstance(node, dict):
        for value in list(node.values()):
            _update_json(value, by_id, fields)
        type_key = next((key for key in TYPE_COLUMNS if key in node), None)
        id_key = next((key for key in ID_COLUMNS if key in node), None)
        if not type_key or not id_key:
            return
        item = by_id.get(str(node.get(id_key) or "").strip())
        if item is None:
            return
        node[type_key] = item.descriptive
        rebuilt: Dict[str, Any] = {}
        for key, value in node.items():
            if key == SYNSET_COLUMN or key in _RETIRED_COLUMNS:
                continue
            rebuilt[key] = value
            if key == type_key:
                rebuilt[SYNSET_COLUMN] = item.synset
        node.clear()
        node.update(rebuilt)
        fields.extend([type_key, SYNSET_COLUMN])
    elif isinstance(node, list):
        for value in node:
            _update_json(value, by_id, fields)


def _insert_synset_column(fieldnames: Sequence[str], type_col: str) -> List[str]:
    drop = {SYNSET_COLUMN, *_RETIRED_COLUMNS}
    fields = [name for name in fieldnames if name not in drop]
    index = fields.index(type_col) + 1
    fields.insert(index, SYNSET_COLUMN)
    return fields


def _assert_ids_unchanged(
    path: Path, id_col: str, original_ids: Sequence[str], original_count: int
) -> None:
    _, rows = read_csv_rows_written(path)
    if len(rows) != original_count:
        raise RuntimeError(f"{path.name}: row count changed ({original_count} -> {len(rows)})")
    new_ids = [(row.get(id_col) or "") for row in rows]
    if list(new_ids) != list(original_ids):
        raise RuntimeError(f"{path.name}: obj ids changed")


def read_csv_rows_written(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def verify_episode(annotations: Path, by_id: Dict[str, ObjectResolution]) -> None:
    undefined = []
    missing_synset = []
    for path in annotation_files(annotations):
        if path.suffix.lower() != ".csv":
            continue
        fieldnames, rows = read_csv_rows_written(path)
        type_col = _type_column(fieldnames)
        id_col = _id_column(fieldnames)
        if not type_col or not id_col:
            continue
        if SYNSET_COLUMN not in fieldnames:
            raise RuntimeError(f"{path.name}: missing {SYNSET_COLUMN}")
        if "obj-category" in fieldnames:
            raise RuntimeError(f"{path.name}: obj-category is still present")
        type_index = fieldnames.index(type_col)
        if fieldnames[type_index + 1] != SYNSET_COLUMN:
            raise RuntimeError(f"{path.name}: {SYNSET_COLUMN} is not next to {type_col}")
        backup = orig_path(path)
        if backup.exists():
            _, orig_rows = _read_exact(backup)
            if len(orig_rows) != len(rows):
                raise RuntimeError(f"{path.name}: row count differs from .orig")
            orig_ids = [(row.get(id_col) or "") for row in orig_rows]
            new_ids = [(row.get(id_col) or "") for row in rows]
            if orig_ids != new_ids:
                raise RuntimeError(f"{path.name}: obj ids differ from .orig")
        for row in rows:
            obj_id = (row.get(id_col) or "").strip()
            if not obj_id or obj_id not in by_id:
                continue
            if is_undefined(row.get(type_col)):
                undefined.append(obj_id)
            if is_undefined(row.get(SYNSET_COLUMN)):
                missing_synset.append(obj_id)
    if undefined:
        raise RuntimeError(
            f"{annotations}: undefined={sorted(set(undefined))[:10]}"
        )
    if missing_synset:
        print(
            f"  no synset for {len(set(missing_synset))} object(s): "
            + ", ".join(sorted(set(missing_synset))[:12])
        )


def _read_exact(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _direct_hypernyms(synset_name: str) -> Set[str]:
    try:
        return {hypernym.name() for hypernym in wn.synset(synset_name).hypernyms()}
    except Exception:
        return set()


def _hypernym_either_way(left: str, right: str) -> bool:
    try:
        return is_hypernym_of(left, right) or is_hypernym_of(right, left)
    except Exception:
        return False


def confusable_pairs(
    resolved: Sequence[ObjectResolution],
) -> List[Tuple[ObjectResolution, ObjectResolution, str]]:
    """Pairs with the same synset, a hypernym link, or a shared direct hypernym."""
    items = [item for item in resolved if item.synset]
    parents: Dict[str, Set[str]] = {}
    pairs: List[Tuple[ObjectResolution, ObjectResolution, str]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.synset == right.synset:
                reason = "same_synset"
            elif _hypernym_either_way(left.synset, right.synset):
                reason = "hypernym"
            else:
                if left.synset not in parents:
                    parents[left.synset] = _direct_hypernyms(left.synset)
                if right.synset not in parents:
                    parents[right.synset] = _direct_hypernyms(right.synset)
                if not (parents[left.synset] & parents[right.synset]):
                    continue
                reason = "shared_direct_hypernym"
            pairs.append((left, right, reason))
    return pairs


def print_confusable(episode: Path, resolved: Sequence[ObjectResolution]) -> None:
    pairs = confusable_pairs(resolved)
    print(f"confusable pairs in {episode.name}: {len(pairs)}")
    print(
        "obj-id-a\tobj-type-a\tobj-synset-a\tobj-id-b\tobj-type-b\tobj-synset-b\treason"
    )
    for left, right, reason in pairs:
        print(
            "\t".join(
                [
                    left.obj_id,
                    left.descriptive,
                    left.synset,
                    right.obj_id,
                    right.descriptive,
                    right.synset,
                    reason,
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", type=Path, required=True, metavar="PATH")
    parser.add_argument(
        "--split",
        choices=HOUSE_SPLITS,
        help="House file split when episode metadata does not name one. "
        "Omit it to keep Objaverse names taken from the obj-id.",
    )
    args = parser.parse_args()
    episodes = find_episodes(args.apply)
    print(f"{len(episodes)} episode(s)")
    apply_episodes(episodes, args.split)


if __name__ == "__main__":
    main()
