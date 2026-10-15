"""Descriptive object type and WordNet synset for one THOR object.

Used at export time and by the offline normalizer. The object id is not rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from nltk.corpus import wordnet2022 as wn

from utils.constants.object_constants import AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET
from utils.objaverse_annotation import get_objaverse_annotations

_UNDEFINED = {"", "undefined", "none", "nan", "null"}


@dataclass(frozen=True)
class ObjectLabel:
    obj_type: str
    synset: str
    source: str  # thor_type | objaverse_annotation | id_name


def is_undefined(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _UNDEFINED


def is_id_placeholder(obj_id: str, value: Any) -> bool:
    """object_state stores the Objaverse id head when THOR type is Undefined."""
    if is_undefined(value):
        return False
    head = str(obj_id).split("|", 1)[0]
    return head.startswith("Obja") and len(head) > 4 and str(value).strip() == head


def lemma_to_camel(lemma: str) -> str:
    parts = re.split(r"[\s_\-]+", lemma.strip())
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def name_from_obj_id(obj_id: str) -> str:
    head = str(obj_id).split("|", 1)[0]
    if head.startswith("Obja") and len(head) > 4:
        return head[4:]
    return head


def first_noun_synset(name: str) -> str:
    """First wordnet2022 noun synset for a fallback name."""
    queries = []
    lowered = re.sub(r"[\s_\-]+", "_", name.strip().lower())
    if lowered:
        queries.append(lowered)
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", name or "")
    if len(parts) > 1:
        joined = "_".join(part.lower() for part in parts)
        if joined not in queries:
            queries.append(joined)
    for query in queries:
        found = wn.synsets(query, pos=wn.NOUN)
        if found:
            return found[0].name()
    return ""


def _annotation(asset_id: Optional[str]) -> Optional[dict]:
    if not asset_id:
        return None
    try:
        table = get_objaverse_annotations()
    except Exception:
        return None
    record = table.get(str(asset_id))
    return record if isinstance(record, dict) else None


def describe_object(object_id: str, object_type: Any, asset_id: Optional[str] = None) -> ObjectLabel:
    """Descriptive name and synset. ``object_id`` is kept as the join key elsewhere."""
    raw_type = "" if object_type is None else str(object_type).strip()
    record = _annotation(asset_id)
    unnamed = is_undefined(raw_type) or is_id_placeholder(object_id, raw_type)

    if unnamed and record is not None:
        lemma = lemma_to_camel(str(record.get("most_specific_lemma") or ""))
        synset = str(record.get("synset") or "")
        if lemma:
            return ObjectLabel(lemma, synset, "objaverse_annotation")

    if unnamed:
        name = name_from_obj_id(object_id)
        return ObjectLabel(name, first_noun_synset(name), "id_name")

    synset = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.get(raw_type, "")
    if not synset and record is not None:
        synset = str(record.get("synset") or "")
    return ObjectLabel(raw_type, synset, "thor_type")
