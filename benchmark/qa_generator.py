import random
import argparse
import json
import numpy as np

"""
Spatial Benchmark Question Generator
Tasks: T1 (visible agent-object), T2 (non-visible agent-object),
       T3 (two visible objects), T4 (N visible objects), T6 (topological)
Input:  one JSON file per trajectory (your graph structure)
Output: questions.json, answers.json, metadata.json
"""
import sys
import json
import uuid
import random
import itertools
from pathlib import Path
sys.path.append("../")
from spatial_data_generation import get_records_objects

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DISTANCE_LABELS  = ["near", "far"]
DIRECTION_AXES   = {0: "lateral", 1: "vertical", 2: "depth"}
DIRECTION_VALUES = {
    "lateral": ["left", "right"],
    "vertical": ["above", "below"],
    "depth":    ["front", "behind"],
}
# Minimum steps since last seen to be a valid T2 question
T2_MIN_STEPS_UNSEEN = 1
T2_MAX_STEPS_UNSEEN = 3

# For T6: receptacle relation means NTPP (inside)
# Touching/adjacent approximated from distance == "near" + on same surface
TOPOLOGICAL_RELATIONS = ["disconnected", "touching", "inside"]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def parse_category(object_id: str) -> str:
    """Extract human-readable category from THOR object ID.
    'Cup|2|10' -> 'Cup'
    Falls back to object_id if no pipe character found.
    """
    return object_id.split("|")[0] if "|" in object_id else object_id


def make_question_id(task: str, template: str, counter: int) -> str:
    return f"{task}-{template}-{counter:04d}"


def get_dominant_relations(relation: list) -> dict:
    """
    relation is [lateral, vertical, depth], e.g. ["right", "", "front"].
    Returns dict of axis -> value for non-empty axes only.
    """
    axes = ["lateral", "vertical", "depth"]
    return {
        axes[i]: relation[i]
        for i in range(3)
        if relation[i] != ""
    }


def edge_lookup(edges: list, source: str, target: str) -> dict | None:
    """Find edge by source and target."""
    for e in edges:
        if e["source"] == source and e["target"] == target:
            return e
    return None


def shuffle_options(correct_label: str, distractors: list) -> tuple:
    """
    Given correct answer label and list of distractor labels,
    return (options_dict, correct_key).
    options_dict: {"A": "...", "B": "...", ...}
    correct_key: "A" | "B" | ...
    """
    all_options = [correct_label] + distractors
    random.shuffle(all_options)
    keys = ["A", "B", "C", "D", "E", "F"]
    options = {keys[i]: all_options[i] for i in range(len(all_options))}
    correct_key = next(k for k, v in options.items() if v == correct_label)
    return options, correct_key


# # ---------------------------------------------------------------------------
# # OBJECT-TO-OBJECT EDGE COMPUTATION (needed for T3, T4)
# # ---------------------------------------------------------------------------

# def compute_object_edges(visible_objects: dict) -> list:
#     """
#     Compute object-to-object spatial relations from local_position.
#     local_position is [lateral, vertical, depth] relative to agent.
#     Relation between A and B from agent perspective:
#         diff = B.local_position - A.local_position
#     """
#     edges = []
#     obj_ids = list(visible_objects.keys())

#     for id_a, id_b in itertools.combinations(obj_ids, 2):
#         obj_a = visible_objects[id_a]
#         obj_b = visible_objects[id_b]

#         if "local_position" not in obj_a or "local_position" not in obj_b:
#             continue

#         lp_a = obj_a["local_position"]   # [lat, vert, depth]
#         lp_b = obj_b["local_position"]

#         diff = [lp_b[i] - lp_a[i] for i in range(3)]

#         thresholds = [0.15, 0.10, 0.15]  # meters, per axis
#         relation = [""] * 3
#         axes_vals = [
#             ("lateral",  ("right", "left")),
#             ("vertical", ("above", "below")),
#             ("depth",    ("front",  "behind")),
#         ]
#         for i, (_, (pos_label, neg_label)) in enumerate(axes_vals):
#             if diff[i] > thresholds[i]:
#                 relation[i] = pos_label
#             elif diff[i] < -thresholds[i]:
#                 relation[i] = neg_label

#         # Euclidean distance between objects
#         dist_m = sum((lp_b[i] - lp_a[i]) ** 2 for i in range(3)) ** 0.5
#         if dist_m < 0.5:
#             dist_label = "near"
#         elif dist_m < 1.2:
#             dist_label = "medium"
#         else:
#             dist_label = "far"

#         edges.append({
#             "source": id_a,
#             "target": id_b,
#             "distance": dist_label,
#             "relation": relation,
#             "computed": True,
#         })

#     return edges


# ---------------------------------------------------------------------------
# T1 — AGENT ↔ VISIBLE OBJECTS
# ---------------------------------------------------------------------------

def generate_T1(step: dict, traj_id: str, scene: str,
                q_counter: dict, actions=None, thresholds=None,
                mov_constant=None, nav_path=None) -> tuple:
    """
    Returns (questions, answers, metadata) lists for this step.
    q_counter is a mutable dict {"n": int} for unique IDs across steps.
    """
    questions, answers, metadata = [], [], []
    step_id   = step["step"]
    image_path = step.get("image_path", "")
    edges     = step.get("edges_visible", [])
    vis_objs  = step.get("visible_objects", {})

    agent_edges = [e for e in edges
                   if e["source"] == "agent" and not e.get("inferred", False)]

    for edge in agent_edges:
        obj_id   = edge["target"]
        category = vis_objs.get(obj_id, {}).get("category",
                                                 parse_category(obj_id))
        relation  = edge.get("angle_relation", ["", "", ""])
        distance_label  = edge.get("distance_label", "")
        distance_metric = edge.get("distance_metric", 0.0)
        dominant  = get_dominant_relations(relation)

        # ── T1-D1: identify direction (single dominant axis) ──────────────
        for axis, value in dominant.items():
            opposites = {
                "left": "right", "right": "left",
                "front": "behind", "behind": "front",
                "above": "below", "below": "above",
            }
            correct = value
            distractors = [opposites[value]]

            # add one or two plausible distractors from other axes
            axis_options = DIRECTION_VALUES[axis]
            for opt in axis_options:
                if opt != value and opt not in distractors:
                    distractors.append(opt)
                    break
            # pad to 3 distractors if needed
            all_dirs = ["left", "right", "front", "behind", "above", "below"]
            for opt in all_dirs:
                if len(distractors) >= 3:
                    break
                if opt != correct and opt not in distractors:
                    distractors.append(opt)

            options, correct_key = shuffle_options(correct, distractors[:3])

            q_counter["n"] += 1
            qid = make_question_id("T1", "D1", q_counter["n"])

            questions.append({
                "question_id":   qid,
                "trajectory_id": traj_id,
                "scene":         scene,
                "step_id":       step_id,
                "image_path":    image_path,
                "task_type":     "T1",
                "template_id":   "T1-D1",
                "object_id":     obj_id,
                "object_type":   category,
                "question": (
                    f"Where is the {category.lower()} relative to "
                    f"your current position and heading?"
                ),
                "options": options,
            })
            answers.append({
                "question_id":  qid,
                "correct_option": correct_key,
                "correct_label":  correct,
            })
            metadata.append({
                "question_id":   qid,
                "trajectory_id": traj_id,
                "step_id":       step_id,
                "object_id":     obj_id,
                "queried_axis":  axis,
                "relation_array": relation,
                "distance_label": distance_label,
                "distance_metric": distance_metric,
                "inferred":      False,
                "axis1_uttal":   "extrinsic-static",
                "axis2_frame":   "egocentric",
                "axis3_scale":   "landmark",
            })

        # ── T1-D2: binary yes/no for a specific direction ─────────────────
        if dominant:
            axis   = random.choice(list(dominant.keys()))
            value  = dominant[axis]
            is_positive = random.random() > 0.5

            if is_positive:
                direction_asked = value
                correct_label   = "Yes"
            else:
                opposites = {
                    "left": "right", "right": "left",
                    "front": "behind", "behind": "front",
                    "above": "below", "below": "above",
                }
                direction_asked = opposites[value]
                correct_label   = "No"

            options, correct_key = shuffle_options(correct_label, ["No" if correct_label == "Yes" else "Yes"])

            q_counter["n"] += 1
            qid = make_question_id("T1", "D2", q_counter["n"])

            questions.append({
                "question_id":   qid,
                "trajectory_id": traj_id,
                "scene":         scene,
                "step_id":       step_id,
                "image_path":    image_path,
                "task_type":     "T1",
                "template_id":   "T1-D2",
                "object_id":     obj_id,
                "object_type":   category,
                "question": (
                    f"Is the {category.lower()} to your {direction_asked}?"
                ),
                "options": {"A": "Yes", "B": "No"},
            })
            answers.append({
                "question_id":    qid,
                "correct_option": correct_key,
                "correct_label":  correct_label,
            })
            metadata.append({
                "question_id":    qid,
                "trajectory_id":  traj_id,
                "step_id":        step_id,
                "object_id":      obj_id,
                "direction_asked": direction_asked,
                "relation_array": relation,
                "is_positive_example": is_positive,
                "axis1_uttal":    "extrinsic-static",
                "axis2_frame":    "egocentric",
                "axis3_scale":    "landmark",
            })

        # # ── T1-M1: qualitative distance ───────────────────────────────────
        # if distance_label in DISTANCE_LABELS:
        #     all_dist = ["near", "far"]
        #     distractors = [d for d in all_dist if d != distance_label]
        #     options, correct_key = shuffle_options(distance_label, distractors)

        #     q_counter["n"] += 1
        #     qid = make_question_id("T1", "M1", q_counter["n"])

        #     questions.append({
        #         "question_id":   qid,
        #         "trajectory_id": traj_id,
        #         "scene":         scene,
        #         "step_id":       step_id,
        #         "image_path":    image_path,
        #         "task_type":     "T1",
        #         "template_id":   "T1-M1",
        #         "object_id":     obj_id,
        #         "object_type":   category,
        #         "question": (
        #             f"How far is the {category.lower()} "
        #             f"from your current position?"
        #         ),
        #         "options": options,
        #     })
        #     answers.append({
        #         "question_id":    qid,
        #         "correct_option": correct_key,
        #         "correct_label":  distance_label,
        #     })
        #     metadata.append({
        #         "question_id":   qid,
        #         "trajectory_id": traj_id,
        #         "step_id":       step_id,
        #         "object_id":     obj_id,
        #         "distance_label": distance_label,
        #         "distance_metric": distance_metric,
        #         "axis1_uttal":   "extrinsic-static",
        #         "axis2_frame":   "egocentric",
        #         "axis3_scale":   "landmark",
        #     })

    # ── T1-M2: which of two objects is closer ─────────────────────────────
    dist_edges = [e for e in agent_edges if e.get("distance_label") in DISTANCE_LABELS]
    # print("dist_edges:", dist_edges)
    for e_a, e_b in itertools.combinations(dist_edges, 2):
        if e_a["distance_label"] == e_b["distance_label"]:
            continue  # skip ambiguous same-distance pairs

        dist_order = {"within reach": 0, "nearby": 1, "visible": 2}
        if dist_order[e_a["distance_label"]] < dist_order[e_b["distance_label"]]:
            closer_id   = e_a["target"]
            closer_cat  = vis_objs.get(closer_id, {}).get("category", parse_category(closer_id))
            farther_id  = e_b["target"]
            farther_cat = vis_objs.get(farther_id, {}).get("category", parse_category(farther_id))
        else:
            closer_id   = e_b["target"]
            closer_cat  = vis_objs.get(closer_id, {}).get("category", parse_category(closer_id))
            farther_id  = e_a["target"]
            farther_cat = vis_objs.get(farther_id, {}).get("category", parse_category(farther_id))

        correct_label = f"The {closer_cat.lower()}"
        distractor    = f"The {farther_cat.lower()}"
        options, correct_key = shuffle_options(correct_label, [distractor])

        q_counter["n"] += 1
        qid = make_question_id("T1", "M2", q_counter["n"])

        questions.append({
            "question_id":   qid,
            "trajectory_id": traj_id,
            "scene":         scene,
            "step_id":       step_id,
            "image_path":    image_path,
            "task_type":     "T1",
            "template_id":   "T1-M2",
            "object_id_a":   closer_id,
            "object_id_b":   farther_id,
            "question": (
                f"Which is closer to you — "
                f"the {closer_cat.lower()} or the {farther_cat.lower()}?"
            ),
            "options": options,
        })
        answers.append({
            "question_id":    qid,
            "correct_option": correct_key,
            "correct_label":  correct_label,
        })
        metadata.append({
            "question_id":    qid,
            "trajectory_id":  traj_id,
            "step_id":        step_id,
            "object_id_a":    closer_id,
            "object_id_b":    farther_id,
            "distance_a":     e_a["distance_label"],
            "distance_b":     e_b["distance_label"],
            "distance_metric_a": e_a["distance_metric"],
            "distance_metric_b": e_b["distance_metric"],
            "axis1_uttal":    "extrinsic-static",
            "axis2_frame":    "egocentric",
            "axis3_scale":    "landmark",
        })

    return questions, answers, metadata


# ---------------------------------------------------------------------------
# T2 — AGENT ↔ NON-VISIBLE OBJECTS
# ---------------------------------------------------------------------------

def get_actions_since(actions: dict, last_seen: int, current_step: int) -> list:
    """Return list of actions taken since last_seen step."""
    res = []
    if not actions:
        return res
    for i in range(last_seen + 1, current_step+1):
        res.append(actions[i])
    return res

def get_action_sequence_text(actions_since: list, degrees: int, mov_constant: float) -> str:
    res = []
    for action in actions_since:
        act_text = action.replace("_", " ")
        if 'left' in act_text or 'right' in act_text:
            act_text += f" ({degrees}°)"
        else:
            act_text += f" ({mov_constant}m)"
        res.append(act_text)
    result = ", ".join(res)
    return result if result else "no actions"

def generate_T2(step: dict, traj_id: str, scene: str,
                q_counter: dict, actions: dict, thresholds: list,
                mov_constant: float, nav_path: str=None) -> tuple:
    questions, answers, metadata = [], [], []
    step_id    = step["step"]
    image_path = step.get("image_path", "")
    inferred   = step.get("edges_inferred", [])
    non_vis    = step.get("non_visible_objects", {})
    degrees    = step.get("degrees", 0)

    for edge in inferred:
        obj_id    = edge["target"]
        last_seen = edge.get("last_seen", None)
        # if last_seen is None:
        #     obj_meta  = non_vis.get(obj_id, {})
        #     last_seen = obj_meta.get("last_seen_step", None)

        if last_seen is None:
            continue
        steps_unseen = step_id - last_seen
        if steps_unseen < T2_MIN_STEPS_UNSEEN:
            continue
        actions_since = get_actions_since(actions, last_seen, step_id)
        action_sequence_text = get_action_sequence_text(actions_since, degrees, mov_constant)

        opposites = {
                "left": "right", "right": "left",
                "front": "behind", "behind": "front",
                "above": "below", "below": "above",
        }

        category = non_vis.get(obj_id, {}).get("category", parse_category(obj_id))
        relation  = edge.get("angle_relation", ["", "", ""])
        distance_label  = edge.get("distance_label", "")
        distance_metric = edge.get("distance_metric", 0.0)
        dominant  = get_dominant_relations(relation)

        for axis, value in dominant.items():
            all_dirs    = ["left", "right", "front", "behind", "above", "below"]
            distractors = [opposites[value]]
            for opt in all_dirs:
                if len(distractors) >= 3:
                    break
                if opt != value and opt not in distractors:
                    distractors.append(opt)

            options, correct_key = shuffle_options(value, distractors[:3])

            q_counter["n"] += 1
            qid = make_question_id("T2", "D1", q_counter["n"])

            questions.append({
                "question_id":   qid,
                "trajectory_id": traj_id,
                "scene":         scene,
                "step_id":       step_id,
                "image_path":    image_path,
                "task_type":     "T2",
                "template_id":   "T2-D1",
                "object_id":     obj_id,
                "object_type":   category,
                "last_seen_step": last_seen,
                "steps_unseen":  steps_unseen,
                "question": (
                    f"The {category.lower()} was last seen {steps_unseen} step(s) ago. "
                    f"Since then the agent performed: {action_sequence_text}. "
                    f"Where is the {category.lower()} relative to your current position?"
                ),
                "options": options,
            })
            answers.append({
                "question_id":    qid,
                "correct_option": correct_key,
                "correct_label":  value,
            })
            metadata.append({
                "question_id":    qid,
                "trajectory_id":  traj_id,
                "step_id":        step_id,
                "object_id":      obj_id,
                "last_seen_step": last_seen,
                "steps_unseen":   steps_unseen,
                "queried_axis":   axis,
                "relation_array": relation,
                "distance_label": distance_label,
                "distance_metric": distance_metric,
                "inferred":       True,
                "axis1_uttal":    "extrinsic-static",
                "axis2_frame":    "egocentric",
                "axis3_scale":    "route",
            })

    return questions, answers, metadata


# ---------------------------------------------------------------------------
# T3 — TWO VISIBLE OBJECTS (object-to-object from agent perspective)
# ---------------------------------------------------------------------------

def generate_T3(step: dict, traj_id: str, scene: str,
                q_counter: dict, actions: dict, thresholds: list,
                mov_constant: float, nav_path: str=None) -> tuple:
    questions, answers, metadata = [], [], []
    step_id    = step["step"]
    image_path = step.get("image_path", "")
    vis_objs   = step.get("visible_objects", {})
    edges      = step.get("edges_visible", [])
    

    # obj_edges = compute_object_edges(vis_objs)
    for edge in edges:
        # print("edge:", edge)
        id_a     = edge["source"]
        if id_a == "agent":
            continue
        id_b     = edge["target"]
        cat_a    = vis_objs.get(id_a, {}).get("category", parse_category(id_a))
        cat_b    = vis_objs.get(id_b, {}).get("category", parse_category(id_b))
        relation = edge["angle_relation"]
        dominant = get_dominant_relations(relation)

        if not dominant:
            continue

        for axis, value in dominant.items():
            # Question: where is B relative to A?
            opposites = {
                "left": "right", "right": "left",
                "front": "behind", "behind": "front",
                "above": "below", "below": "above",
            }
            all_dirs    = ["left", "right", "front", "behind", "above", "below"]
            distractors = [opposites[value]]
            for opt in all_dirs:
                if len(distractors) >= 3:
                    break
                if opt != value and opt not in distractors:
                    distractors.append(opt)

            options, correct_key = shuffle_options(value, distractors[:3])

            q_counter["n"] += 1
            qid = make_question_id("T3", "D1", q_counter["n"])

            questions.append({
                "question_id":   qid,
                "trajectory_id": traj_id,
                "scene":         scene,
                "step_id":       step_id,
                "image_path":    image_path,
                "task_type":     "T3",
                "template_id":   "T3-D1",
                "object_id_a":   id_a,
                "object_id_b":   id_b,
                "object_type_a": cat_a,
                "object_type_b": cat_b,
                "question": (
                    f"From your current viewpoint, where is "
                    f"the {cat_b.lower()} relative to "
                    f"the {cat_a.lower()}?"
                ),
                "options": options,
            })
            answers.append({
                "question_id":    qid,
                "correct_option": correct_key,
                "correct_label":  value,
            })
            metadata.append({
                "question_id":    qid,
                "trajectory_id":  traj_id,
                "step_id":        step_id,
                "object_id_a":    id_a,
                "object_id_b":    id_b,
                "queried_axis":   axis,
                "relation_array": relation,
                "distance_label": edge["distance_label"],
                "distance_metric": edge["distance_metric"],
                "axis1_uttal":    "extrinsic-static",
                "axis2_frame":    "egocentric",
                "axis3_scale":    "landmark",
            })

    return questions, answers, metadata


# ---------------------------------------------------------------------------
# T4 — N VISIBLE OBJECTS (ordering / multi-object)
# ---------------------------------------------------------------------------

def generate_T4(step: dict, traj_id: str, scene: str,
                q_counter: dict, actions: dict, thresholds: list,
                mov_constant: float, nav_path: str=None) -> tuple:
    questions, answers, metadata = [], [], []
    step_id    = step["step"]
    image_path = step.get("image_path", "")
    vis_objs   = step.get("visible_objects", {})
    edges      = step.get("edges_visible", [])

    if len(vis_objs) < 3:
        return questions, answers, metadata

    agent_edges = [e for e in edges
                   if e["source"] == "agent"
                   and not e.get("inferred", False)
                   and e.get("distance_label") in DISTANCE_LABELS]

    if len(agent_edges) < 3:
        return questions, answers, metadata

    # ── T4-M3: rank 3 objects nearest to farthest ────────────────────────
    dist_order = {"whithin reach": 0, "nearby": 1, "visible": 2}
    # sample triplets
    for triple in itertools.combinations(agent_edges, 3):
        dists = [dist_order[e["distance_label"]] for e in triple]
        # skip if any two have same distance (ambiguous ordering)
        if len(set(dists)) < 3:
            continue

        sorted_triple = sorted(triple, key=lambda e: dist_order[e["distance_label"]])
        cats = [
            vis_objs.get(e["target"], {}).get("category", parse_category(e["target"]))
            for e in sorted_triple
        ]

        correct_label = f"{cats[0]}, {cats[1]}, {cats[2]}"

        # generate all permutations as options (max 6)
        all_perms = list(itertools.permutations(cats))
        distractor_labels = [
            f"{p[0]}, {p[1]}, {p[2]}"
            for p in all_perms
            if list(p) != cats
        ]
        random.shuffle(distractor_labels)
        distractors = distractor_labels[:5]

        options, correct_key = shuffle_options(correct_label, distractors)

        q_counter["n"] += 1
        qid = make_question_id("T4", "M3", q_counter["n"])

        questions.append({
            "question_id":   qid,
            "trajectory_id": traj_id,
            "scene":         scene,
            "step_id":       step_id,
            "image_path":    image_path,
            "task_type":     "T4",
            "template_id":   "T4-M3",
            "object_ids":    [e["target"] for e in sorted_triple],
            "object_types":  cats,
            "question": (
                f"Order the following objects from nearest to farthest "
                f"from your current position: "
                f"{cats[0]}, {cats[1]}, {cats[2]}."
            ),
            "options": options,
        })
        answers.append({
            "question_id":    qid,
            "correct_option": correct_key,
            "correct_label":  correct_label,
        })
        metadata.append({
            "question_id":   qid,
            "trajectory_id": traj_id,
            "step_id":       step_id,
            "object_ids":    [e["target"] for e in sorted_triple],
            "distances":     [e["distance_label"] for e in sorted_triple],
            "axis1_uttal":   "extrinsic-static",
            "axis2_frame":   "egocentric",
            "axis3_scale":   "landmark",
        })

        # limit to first valid triplet per step to avoid explosion
        # break

    return questions, answers, metadata


# ---------------------------------------------------------------------------
# T6 — TOPOLOGICAL RELATIONS (DC, EC, NTPP via receptacleObjectIds)
# ---------------------------------------------------------------------------

def bbox_gap(bbox_a: list, bbox_b: list) -> list:
    # bbox format: {"center": [x,y,z], "size": [sx,sy,sz]}
    # gap on each axis = max(0, |center_diff| - (size_a + size_b) / 2)
    gaps = []
    for i in range(3):
        center_diff = abs(bbox_a["center"][i] - bbox_b["center"][i])
        half_sum = (bbox_a["size"][i] + bbox_b["size"][i]) / 2
        gaps.append(max(0, center_diff - half_sum))
    return gaps

def generate_T6(step: dict, traj_id: str, scene: str,
                q_counter: dict, actions: dict, thresholds: list,
                mov_constant=float, obj_path: str=None) -> tuple:
    """
    T6 uses receptacle information to determine topological relations.
    raw_step should contain the full THOR metadata with receptacleObjectIds.
    For V1 we cover:
      NTPP  — object is inside a receptacle
      EC    — object is near (distance==near) and on same surface (approx)
      DC    — object is not inside any receptacle and not near another object
    """
    questions, answers, metadata = [], [], []
    step_id    = step["step"]
    image_path = step.get("image_path", "")
    vis_objs   = step.get("visible_objects", {})
    edges      = step.get("edges_visible", [])
    obj_dict   = get_records_objects(obj_path) if obj_path else {}

    for obj_id, obj_data in vis_objs.items():
        category = obj_data.get("category", parse_category(obj_id))

        # Determine topological relation for each visible object pair
        for other_id, other_data in vis_objs.items():
            if other_id == obj_id:
                continue
            other_cat = other_data.get("category", parse_category(other_id))

            # NTPP: obj_id is inside other_id (receptacle)
            other_receptacles = obj_dict[other_id].get('receptacleObjectIds', [])
            # other_receptacles = other_data.get("receptacleObjectIds", [])
            if obj_id in other_receptacles:
                correct_label = "inside"
                distractors   = ["touching", "separate from"]
                options, correct_key = shuffle_options(correct_label, distractors)

                q_counter["n"] += 1
                qid = make_question_id("T6", "NTPP", q_counter["n"])

                questions.append({
                    "question_id":   qid,
                    "trajectory_id": traj_id,
                    "scene":         scene,
                    "step_id":       step_id,
                    "image_path":    image_path,
                    "task_type":     "T6",
                    "template_id":   "T6-NTPP",
                    "object_id_a":   obj_id,
                    "object_id_b":   other_id,
                    "object_type_a": category,
                    "object_type_b": other_cat,
                    "question": (
                        f"What is the topological relation between "
                        f"the {category.lower()} and "
                        f"the {other_cat.lower()}?"
                    ),
                    "options": options,
                })
                answers.append({
                    "question_id":    qid,
                    "correct_option": correct_key,
                    "correct_label":  correct_label,
                    "rcc8_relation":  "NTPP",
                })
                metadata.append({
                    "question_id":   qid,
                    "trajectory_id": traj_id,
                    "step_id":       step_id,
                    "object_id_a":   obj_id,
                    "object_id_b":   other_id,
                    "rcc8":          "NTPP",
                    "axis1_uttal":   "extrinsic-static",
                    "axis2_frame":   "allocentric",
                    "axis3_scale":   "landmark",
                })
                continue

            # EC: both visible, and they are externally connected / touching
            edge = edge_lookup(edges, "agent", obj_id)
            other_edge = edge_lookup(edges, "agent", other_id)
            obj_in_other = obj_id in obj_dict[other_id].get("receptacleObjectIds", [])  # other_data.get("receptacleObjectIds", [])
            other_in_obj = other_id in obj_dict[obj_id].get("receptacleObjectIds", [])  # obj_data.get("receptacleObjectIds", [])

            if (edge and other_edge
                    and edge.get("distance_label") == "within reach"
                    and other_edge.get("distance_label") == "within reach"
                    and not obj_in_other and not other_in_obj):
                correct_label = "touching"
                distractors   = ["inside", "separate from"]
                options, correct_key = shuffle_options(correct_label, distractors)

                q_counter["n"] += 1
                qid = make_question_id("T6", "EC", q_counter["n"])

                questions.append({
                    "question_id":   qid,
                    "trajectory_id": traj_id,
                    "scene":         scene,
                    "step_id":       step_id,
                    "image_path":    image_path,
                    "task_type":     "T6",
                    "template_id":   "T6-EC",
                    "object_id_a":   obj_id,
                    "object_id_b":   other_id,
                    "object_type_a": category,
                    "object_type_b": other_cat,
                    "question": (
                        f"What is the spatial relationship between "
                        f"the {category.lower()} and "
                        f"the {other_cat.lower()}?"
                    ),
                    "options": options,
                })
                answers.append({
                    "question_id":    qid,
                    "correct_option": correct_key,
                    "correct_label":  correct_label,
                    "rcc8_relation":  "EC",
                })
                metadata.append({
                    "question_id":   qid,
                    "trajectory_id": traj_id,
                    "step_id":       step_id,
                    "object_id_a":   obj_id,
                    "object_id_b":   other_id,
                    "rcc8":          "EC",
                    "axis1_uttal":   "extrinsic-static",
                    "axis2_frame":   "allocentric",
                    "axis3_scale":   "landmark",
                })

    return questions, answers, metadata


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def generate_benchmark(input_path: str, output_dir: str,
                        trajectory_id: str = None, tuple_task: tuple = (),
                        obj_path: str = "") -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    i_task, generator = tuple_task

    with open(input_path, "r") as f:
        data = json.load(f)

    scene   = data.get("scene", "unknown")
    traj_id = trajectory_id or f"{scene}_traj_001"
    thresholds = data.get("thresholds", {})
    mov_constant = data.get("movement_constant", 0.2)
    steps   = data.get("steps", [])

    all_questions, all_answers, all_metadata = [], [], []
    q_counter = {"n": 0}
    actions = {}

    for step in steps:
        actions[step.get("step")] = step.get("action", "unknown")
        # for generator in [generate_T1, generate_T2]: #,
                          #generate_T3, generate_T4, generate_T6]:
        qs, ans, meta = generator(step, traj_id, scene, q_counter, actions, thresholds, mov_constant, obj_path)
        all_questions.extend(qs)
        all_answers.extend(ans)
        all_metadata.extend(meta)
        # break  # for demo, only process first step

    stem = input_path.stem
    with open(output_dir / f"{stem}_questions_{i_task}.json", "w") as f:
        json.dump(all_questions, f, indent=2)
    with open(output_dir / f"{stem}_answers_{i_task}.json", "w") as f:
        json.dump(all_answers, f, indent=2)
    with open(output_dir / f"{stem}_metadata_{i_task}.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    # summary
    from collections import Counter
    task_counts = Counter(q["task_type"] for q in all_questions)
    print(f"Generated {len(all_questions)} questions for trajectory: {traj_id}")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print(f"Output written to: {output_dir}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python qa_generator.py <input.json> <output_dir> [trajectory_id]")
        sys.exit(1)

    input_json   = sys.argv[1]
    output_dir   = sys.argv[2]
    traj_id      = sys.argv[3] if len(sys.argv) > 3 else None
    index        = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    nav_path     = sys.argv[5] if len(sys.argv) > 5 else None
    tasks = [(1, generate_T1), (2, generate_T2), (3, generate_T3), (4, generate_T4), (6, generate_T6)]
    generate_benchmark(input_json, output_dir, traj_id, tasks[index], nav_path)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate QA pairs from episodes")
#     parser.add_argument('--input', type=str, required=True,
#                         help="Path to input JSONL file with episodes")
#     parser.add_argument('--output', type=str, required=True,
#                         help="Path to output JSONL file for QA pairs")
#     return parser.parse_args()

# def main(args):
#     input = args.input
#     output = args.output

#     with open(input, 'r', encoding='utf-8') as f_in:
#         episode = json.load(f_in)

#     if episode:
#         qas = generate_benchmark(episode)
#         data = {"questions": qas}
#         with open(output, 'w', encoding='utf-8') as f_out:
#             # for qa in qas:
#             #     f_out.write(json.dumps(qa) + '\n')
#             json.dump(data, f_out, indent=4)

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)