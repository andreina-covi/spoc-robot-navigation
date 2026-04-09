import random
import argparse
import json
import numpy as np

AXIS_INDEX = {'horizontal': 0, 'vertical': 1, 'depth': 2}
OPPOSITES  = {
    'left': 'right',   'right': 'left',
    'front': 'behind', 'behind': 'front',
    'above': 'below',  'below': 'above',
    'close': 'far',    'far': 'close',
    'medium': 'far'
}
ALL_OPTIONS = {
    'horizontal': ['left', 'right'],
    'vertical':   ['above', 'below'],
    'depth':      ['front', 'behind'],
    'distance':   ['close', 'medium', 'far']
}

MAX_COMPARE_WINDOW = 5   # only compare with the last 5 steps

# Core data structure for a question-answer pair, with all relevant metadata.
def make_qa(
    question, answer, format, q_type,
    axis, step, scene,
    edge=None, metadata=None
):
    return {
        # ── content ────────────────────────────────────
        'question':   question,
        'answer':     answer,

        # ── classification ─────────────────────────────
        'format':     format,     # binary | mcq | open | comparative
        'q_type':     q_type,     # single_step | action_conditioned
                                  # topology | comparative_steps
        'axis':       axis,       # horizontal | vertical | depth | distance

        # ── provenance ─────────────────────────────────
        # 'episode_id': episode_id,
        'scene':      scene,
        'step':       step,       # int or dict {'t': t, 't1': t1}

        # ── source edge ────────────────────────────────
        'edge':       edge,       # the dict from edges_visible/edges_inferred

        # ── extra metadata ─────────────────────────────
        'metadata':   metadata or {}
    }

def generate_single_step(step, edge, axis_name, scene):  # episode_id,
    """
    Generate binary + MCQ + open questions for one edge at one step.
    axis_name: 'horizontal' | 'vertical' | 'depth'
    """
    qas = []

    idx      = AXIS_INDEX[axis_name]
    relation = edge['relation'][idx]

    if not relation:
        return qas   # ambiguous — skip

    source   = edge['source'].split('|')[0]
    target   = edge['target']
    category = target.split('|')[0]

    # ── BINARY ─────────────────────────────────────────────────────────
    # positive (answer: yes)
    q_pos = (f"Is the {category} to the {relation} of the {source}?"
             if axis_name != 'vertical'
             else f"Is the {category} {relation} the {source}?")

    qas.append(make_qa(
        question   = q_pos,
        answer     = 'yes',
        format     = 'binary',
        q_type     = 'single_step',
        axis       = axis_name,
        step       = step['step'],
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge,
        metadata   = {'polarity': 'positive'}
    ))

    # negative (answer: no) — flip the relation
    wrong = OPPOSITES.get(relation)
    if wrong:
        q_neg = (f"Is the {category} to the {wrong} of the {source}?"
                 if axis_name != 'vertical'
                 else f"Is the {category} {wrong} the {source}?")

        qas.append(make_qa(
            question   = q_neg,
            answer     = 'no',
            format     = 'binary',
            q_type     = 'single_step',
            axis       = axis_name,
            step       = step['step'],
            # episode_id = episode_id,
            scene      = scene,
            edge       = edge,
            metadata   = {'polarity': 'negative'}
        ))

    # ── MCQ ────────────────────────────────────────────────────────────
    options    = ALL_OPTIONS[axis_name]
    distractors = [o for o in options if o != relation]

    # build choices dict with correct answer always shuffled
    choices_list = [relation] + distractors
    random.shuffle(choices_list)
    labels   = ['A', 'B', 'C', 'D'][:len(choices_list)]
    choices  = dict(zip(labels, choices_list))
    correct_label = [k for k, v in choices.items() if v == relation][0]

    choices_str = '  '.join([f"{k}) {v}" for k, v in choices.items()])
    q_mcq = (f"Where is the {category} relative to the {source} "
             f"on the {axis_name} axis? {choices_str}")

    qas.append(make_qa(
        question   = q_mcq,
        answer     = correct_label,
        format     = 'mcq',
        q_type     = 'single_step',
        axis       = axis_name,
        step       = step['step'],
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge,
        metadata   = {'choices': choices, 'correct_label': correct_label}
    ))

    # ── OPEN ───────────────────────────────────────────────────────────
    q_open = (f"Where is the {category} relative to the {source} "
              f"on the {axis_name} axis?")

    qas.append(make_qa(
        question   = q_open,
        answer     = relation,
        format     = 'open',
        q_type     = 'single_step',
        axis       = axis_name,
        step       = step['step'],
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge,
        metadata   = {}
    ))

    return qas

def generate_action_conditioned(step_t, step_t1, obj_id,
                                 axis_name, scene):         # episode_id,
    """
    'After [action], is the [object] now [relation] of the agent?'
    Requires object visible at both step_t and step_t1.
    """
    qas = []

    if (obj_id not in step_t['visible_objects'] or
        obj_id not in step_t1['visible_objects']):
        return qas

    idx      = AXIS_INDEX[axis_name]
    action   = step_t1['action']
    degrees = step_t1.get('degrees', 0)
    action_desc = describe_action(action, degrees)
    category = step_t1['visible_objects'][obj_id]['category']

    # relation BEFORE action (step t)
    edge_t = next(
        (e for e in step_t['edges_visible']
         if e['target'] == obj_id and e['source'] == 'agent'),
        None
    )
    # relation AFTER action (step t+1)
    edge_t1 = next(
        (e for e in step_t1['edges_visible']
         if e['target'] == obj_id and e['source'] == 'agent'),
        None
    )

    if not edge_t or not edge_t1:
        return qas

    rel_before = edge_t['relation'][idx]
    rel_after  = edge_t1['relation'][idx]

    if not rel_after:
        return qas   # ambiguous after action — skip

    # single-step displacement for metadata
    displacement = compute_cumulative_displacement(
        [step_t, step_t1],
        from_step = 0,
        to_step   = 1
    )

    # ── BINARY ─────────────────────────────────────────────────────────
    q_pos = (f"After the agent '{action_desc}', "
             f"is the {category} to the {rel_after} of the agent?")

    metadata = {
        'action':      action,
        'degrees':     degrees,
        'rel_before':  rel_before,
        'rel_after':   rel_after,
        'changed':     rel_before != rel_after,
        'displacement': displacement
    }

    qas.append(make_qa(
        question   = q_pos,
        answer     = 'yes',
        format     = 'binary',
        q_type     = 'action_conditioned',
        axis       = axis_name,
        step       = {'t': step_t['step'], 't1': step_t1['step']},
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge_t1,
        metadata   = metadata
    ))

    # negative version
    wrong = OPPOSITES.get(rel_after)
    if wrong:
        q_neg = (f"After the agent performed '{action}', "
                 f"is the {category} to the {wrong} of the agent?")

        qas.append(make_qa(
            question   = q_neg,
            answer     = 'no',
            format     = 'binary',
            q_type     = 'action_conditioned',
            axis       = axis_name,
            step       = {'t': step_t['step'], 't1': step_t1['step']},
            # episode_id = episode_id,
            scene      = scene,
            edge       = edge_t1,
            metadata   = {
                'action':     action,
                'rel_before': rel_before,
                'rel_after':  rel_after,
                'polarity':   'negative'
            }
        ))

    # ── MCQ ────────────────────────────────────────────────────────────
    options      = ALL_OPTIONS[axis_name]
    choices_list = [rel_after] + [o for o in options if o != rel_after]
    random.shuffle(choices_list)
    labels       = ['A', 'B', 'C', 'D'][:len(choices_list)]
    choices      = dict(zip(labels, choices_list))
    correct_label = [k for k, v in choices.items() if v == rel_after][0]
    choices_str   = '  '.join([f"{k}) {v}" for k, v in choices.items()])

    q_mcq = (f"After the agent performed '{action}', where is the "
             f"{category} relative to the agent on the {axis_name} axis? "
             f"{choices_str}")

    qas.append(make_qa(
        question   = q_mcq,
        answer     = correct_label,
        format     = 'mcq',
        q_type     = 'action_conditioned',
        axis       = axis_name,
        step       = {'t': step_t['step'], 't1': step_t1['step']},
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge_t1,
        metadata   = {'choices': choices, 'action': action}
    ))

    # ── OPEN ───────────────────────────────────────────────────────────
    q_open = (f"After the agent performed '{action}', where is the "
              f"{category} relative to the agent on the {axis_name} axis?")

    qas.append(make_qa(
        question   = q_open,
        answer     = rel_after,
        format     = 'open',
        q_type     = 'action_conditioned',
        axis       = axis_name,
        step       = {'t': step_t['step'], 't1': step_t1['step']},
        # episode_id = episode_id,
        scene      = scene,
        edge       = edge_t1,
        metadata   = {'action': action}
    ))

    return qas

def rotation_difficulty(cumulative_degrees):
    """
    Classify topology question difficulty based on
    total rotation since object was last visible.
    """
    abs_rot = abs(cumulative_degrees) % 360  # normalize to 0-360
    difficulty = None

    if abs_rot <= 30:
        difficulty = 'easy'      # small rotation — object roughly same position
    elif abs_rot <= 90:
        difficulty = 'medium'    # quarter turn — significant shift
    elif abs_rot <= 180:
        difficulty = 'hard'      # half turn — object likely behind agent
    else:
        difficulty = 'very_hard' # more than half turn — full spatial reversal
    
    return difficulty

def generate_topology(step_current, inferred_edge,
                       axis_name, episode_steps, scene):        # episode_id,
    """
    'After N actions, where is the [object] that is no longer visible?'
    Uses edges_inferred — your core benchmark contribution.
    """
    qas = []

    idx      = AXIS_INDEX[axis_name]
    relation = inferred_edge['relation'][idx]

    if not relation:
        return qas

    # obj_id         = inferred_edge['target']
    category       = inferred_edge['target'].split('|')[0]
    last_seen_step = inferred_edge['last_seen']
    current_step   = step_current['step']
    n_steps        = current_step - last_seen_step

    if n_steps > MAX_COMPARE_WINDOW:
        return qas # too many steps → question is too hard and less relevant for real-time reasoning
    
    # compute cumulative displacement since object was last seen
    displacement = compute_cumulative_displacement(
        episode_steps, last_seen_step, current_step
    )
    cumulative_rot = displacement['rotation']
    distance_moved = displacement['distance_moved']
    difficulty = rotation_difficulty(cumulative_rot)

    # build readable action sequence WITH degrees
    action_sequence_parts = []
    for i in range(last_seen_step + 1, current_step + 1):
        step   = episode_steps[i]
        action = step['action']
        degrees = step.get('degrees', 0)
        if action is not None:
            desc = describe_action(action, degrees)
            action_sequence_parts.append(desc)

    action_sequence = ', then '.join(action_sequence_parts)

    # add rotation summary for clarity
    if cumulative_rot != 0:
        direction  = "right" if cumulative_rot > 0 else "left"
        rot_summary = (f"(total rotation: {abs(cumulative_rot):.1f}° "
                       f"to the {direction})")
    else:
        rot_summary = "no net rotation"
    
    if distance_moved > 0.05:   # threshold to ignore negligible movement
        move_summary = f"total displacement: {distance_moved:.2f}m"
    else:
        move_summary = "no significant movement"

    context_summary = f"({rot_summary}, {move_summary})"

    base_metadata = {
        'n_steps':             n_steps,
        'last_seen_step':      last_seen_step,
        # 'cumulative_rotation': cumulative_rot,
        # 'distance_moved':      distance_moved,
        'displacement':        displacement,
        'difficulty':          difficulty,
        'action_sequence':     action_sequence_parts,
    }

    # ── BINARY ─────────────────────────────────────────────────────────
    q_pos = (
        f"The agent last saw the {category} {n_steps} step(s) ago. "
        f"Since then, the agent: {action_sequence} {context_summary}. "
        f"Is the {category} now to the {relation} of the agent?"
    )

    qas.append(make_qa(
        question   = q_pos,
        answer     = 'yes',
        format     = 'binary',
        q_type     = 'topology',
        axis       = axis_name,
        step       = current_step,
        # episode_id = episode_id,
        scene      = scene,
        edge       = inferred_edge,
        metadata   = {**base_metadata, 'polarity': 'positive'}
    ))

    # negative version
    wrong = OPPOSITES.get(relation)
    if wrong:
        q_neg = (
            f"The agent last saw the {category} {n_steps} step(s) ago. "
            f"Since then, the agent: {action_sequence} {context_summary}. "
            f"Is the {category} now to the {wrong} of the agent?"
        )

        qas.append(make_qa(
            question   = q_neg,
            answer     = 'no',
            format     = 'binary',
            q_type     = 'topology',
            axis       = axis_name,
            step       = current_step,
            # episode_id = episode_id,
            scene      = scene,
            edge       = inferred_edge,
            metadata   = {**base_metadata, 'polarity': 'negative'}
        ))

    # ── MCQ ────────────────────────────────────────────────────────────
    options      = ALL_OPTIONS[axis_name]
    choices_list = [relation] + [o for o in options if o != relation]
    random.shuffle(choices_list)
    labels       = ['A', 'B', 'C', 'D'][:len(choices_list)]
    choices      = dict(zip(labels, choices_list))
    correct_label = [k for k, v in choices.items() if v == relation][0]
    choices_str   = '  '.join([f"{k}) {v}" for k, v in choices.items()])

    q_mcq = (
        f"The agent last saw the {category} {n_steps} step(s) ago. "
        f"Since then, the agent: {action_sequence} {context_summary}. "
        f"Where is the {category} now relative to the agent "
        f"on the {axis_name} axis? {choices_str}"
    )

    qas.append(make_qa(
        question   = q_mcq,
        answer     = correct_label,
        format     = 'mcq',
        q_type     = 'topology',
        axis       = axis_name,
        step       = current_step,
        # episode_id = episode_id,
        scene      = scene,
        edge       = inferred_edge,
        metadata   = {**base_metadata,
                      'choices': choices,
                      'correct_label': correct_label}
    ))

    # ── OPEN ───────────────────────────────────────────────────────────
    q_open = (
        f"The agent last saw the {category} {n_steps} step(s) ago. "
        f"Since then, the agent: {action_sequence} {context_summary}. "
        f"Where is the {category} now relative to the agent "
        f"on the {axis_name} axis?"
    )

    qas.append(make_qa(
        question   = q_open,
        answer     = relation,
        format     = 'open',
        q_type     = 'topology',
        axis       = axis_name,
        step       = current_step,
        # episode_id = episode_id,
        scene      = scene,
        edge       = inferred_edge,
        metadata   = base_metadata
    ))

    return qas

def generate_comparative_steps(step_t, step_t1, obj_id, scene):    # episode_id
    """
    'Was the [object] closer to the agent at step t or step t+n?'
    Uses distance labels from edges at two different steps.
    """
    qas = []

    # get distance at step t
    edge_t = next(
        (e for e in step_t['edges_visible']
         if e['target'] == obj_id and e['source'] == 'agent'),
        None
    )
    # get distance at step t1 (visible or inferred)
    edge_t1 = next(
        (e for e in step_t1['edges_visible'] + step_t1['edges_inferred']
         if e['target'] == obj_id and e['source'] == 'agent'),
        None
    )

    if not edge_t or not edge_t1:
        return qas

    dist_t  = edge_t['distance']
    dist_t1 = edge_t1['distance']

    if not dist_t or not dist_t1:
        return qas

    # only generate if distances are actually different
    # (same distance → question is trivial)
    if dist_t == dist_t1:
        return qas

    order    = {'close': 0, 'medium': 1, 'far': 2}
    category = (step_t['visible_objects'].get(obj_id, {}).get('category')
                or obj_id.split('|')[0])

    # which step had the object closer?
    if order[dist_t] < order[dist_t1]:
        closer_step  = step_t['step']
        farther_step = step_t1['step']
    else:
        closer_step  = step_t1['step']
        farther_step = step_t['step']

    n_steps = step_t1['step'] - step_t['step']

    # ── COMPARATIVE BINARY ─────────────────────────────────────────────
    q_pos = (
        f"Was the {category} closer to the agent at step {step_t['step']} "
        f"than it was {n_steps} step(s) later?"
    )
    answer_pos = 'yes' if closer_step == step_t['step'] else 'no'

    qas.append(make_qa(
        question   = q_pos,
        answer     = answer_pos,
        format     = 'comparative',
        q_type     = 'comparative_steps',
        axis       = 'distance',
        step       = {'t': step_t['step'], 't1': step_t1['step']},
        # episode_id = episode_id,
        scene      = scene,
        edge       = None,
        metadata   = {
            'dist_t':      dist_t,
            'dist_t1':     dist_t1,
            'closer_step': closer_step,
            'n_steps':     n_steps
        }
    ))

    # ── COMPARATIVE MCQ ────────────────────────────────────────────────
    q_mcq = (
        f"The {category} was {dist_t} from the agent at step {step_t['step']} "
        f"and {dist_t1} from the agent {n_steps} step(s) later. "
        f"At which step was the {category} closer to the agent? "
        f"A) step {step_t['step']}  B) step {step_t1['step']}"
    )
    answer_mcq = 'A' if closer_step == step_t['step'] else 'B'

    qas.append(make_qa(
        question   = q_mcq,
        answer     = answer_mcq,
        format     = 'comparative',
        q_type     = 'comparative_steps',
        axis       = 'distance',
        step       = {'t': step_t['step'], 't1': step_t1['step']},
        # episode_id = episode_id,
        scene      = scene,
        edge       = None,
        metadata   = {
            'dist_t':  dist_t,
            'dist_t1': dist_t1,
            'n_steps': n_steps
        }
    ))

    return qas

def compute_cumulative_displacement(episode_steps, from_step, to_step):
    """
    Track both rotation and translation since from_step.
    """

    cumulative_rotation    = 0.0
    cumulative_translation = np.array([0.0, 0.0, 0.0])

    for i in range(from_step + 1, to_step + 1):
        step    = episode_steps[i]
        action  = step['action']
        degrees = step.get('degrees', 0)

        # rotation
        if action == 'rotate_right':
            cumulative_rotation += degrees
        elif action == 'rotate_left':
            cumulative_rotation -= degrees

        # translation — use agent position difference
        if i > from_step:
            pos_prev = np.array(episode_steps[i-1]['agent']['position'])
            pos_curr = np.array(step['agent']['position'])
            cumulative_translation += (pos_curr - pos_prev)

    return {
        'rotation':    np.round(cumulative_rotation, 3),
        # 'translation': np.round(cumulative_translation, 3).tolist(),
        'distance_moved': np.round(float(np.linalg.norm(
                               cumulative_translation[[0, 2]]  # XZ only
                           )), 3)
    }

def generate_all_qas(episode):
    all_qas    = []
    steps      = episode['steps']
    # episode_id = episode['episode_id']
    scene      = episode['scene']
    axes       = ['horizontal', 'vertical', 'depth']

    for i, step in enumerate(steps):
        # ── 1. Single-step ─────────────────────────────────────────────
        for edge in step['edges_visible']:
            for axis in axes:
                qas = generate_single_step(
                    step, edge, axis, scene
                )
                all_qas.extend(qas)

        # ── 2. Action-conditioned ──────────────────────────────────────
        if i > 0:
            step_prev      = steps[i - 1]
            common_objects = (
                set(step['visible_objects'].keys()) &
                set(step_prev['visible_objects'].keys())
            )
            for obj_id in common_objects:
                for axis in axes:
                    qas = generate_action_conditioned(
                        step_prev, step, obj_id, axis, scene
                    )
                    all_qas.extend(qas)

        # ── 3. N-step topology ─────────────────────────────────────────
        for inferred_edge in step['edges_inferred']:
            for axis in axes:
                qas = generate_topology(
                    step, inferred_edge, axis, steps, scene
                )
                all_qas.extend(qas)

        # ── 4. Comparative across steps ────────────────────────────────
        # compare current step with all previous steps
        for j in range(max(0, i - MAX_COMPARE_WINDOW), i):
            step_prev = steps[j]
            # objects that were visible at step j
            for obj_id in step_prev['visible_objects']:
                qas = generate_comparative_steps(
                    step_prev, step, obj_id, scene
                )
                all_qas.extend(qas)

    return all_qas

def compute_cumulative_rotation(episode_steps, from_step, to_step):
    """
    Compute the total rotation accumulated by the agent
    between from_step and to_step (inclusive of actions at each step).

    Returns cumulative degrees (signed: positive = right, negative = left)
    and a structured breakdown per step.
    """
    cumulative = 0.0
    breakdown  = []

    for i in range(from_step + 1, to_step + 1):
        step   = episode_steps[i]
        action = step['action']
        degrees = step.get('degrees', 0)

        # signed rotation: right = positive, left = negative
        if action in ('rotate_right',):
            delta = +degrees
        elif action in ('rotate_left',):
            delta = -degrees
        else:
            delta = 0   # MoveAhead, MoveBack, LookUp, LookDown
                        # do not change horizontal facing

        cumulative += delta
        breakdown.append({
            'step':   i,
            'action': action,
            'degrees': degrees,
            'delta':   delta,
            'cumulative_so_far': cumulative
        })

    return cumulative, breakdown

def describe_action(action, degrees):
    """
    Convert action + degrees into a human-readable description.
    """
    templates = {
        'rotate_right': f"rotated {degrees}° to the right",
        'rotate_left':  f"rotated {degrees}° to the left",
        'move_ahead':   "moved forward",
        'move_back':    "moved backward",
        'look_up':      f"tilted the camera up {degrees}°",
        'look_down':    f"tilted the camera down {degrees}°",
    }
    return templates.get(action, f"performed '{action}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA pairs from episodes")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to input JSONL file with episodes")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output JSONL file for QA pairs")
    return parser.parse_args()

def main(args):
    input = args.input
    output = args.output

    with open(input, 'r', encoding='utf-8') as f_in:
        episode = json.load(f_in)

    if episode:
        qas = generate_all_qas(episode)
        data = {"questions": qas}
        with open(output, 'w', encoding='utf-8') as f_out:
            # for qa in qas:
            #     f_out.write(json.dumps(qa) + '\n')
            json.dump(data, f_out, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)