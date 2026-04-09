from collections import defaultdict
import random

def qa_single_step_relation(step, edge, axis):
    """
    Generate a binary or MCQ question about a visible relation at one step.
    
    Parameters
    ----------
    step  : dict  — one step from your episode
    edge  : dict  — one edge from step['edges_visible']
    axis  : int   — 0=horizontal, 1=vertical, 2=depth
    """
    axis_names  = ['horizontal', 'vertical', 'depth']
    axis_labels = [
        ['left', 'right'],
        ['above', 'below'],
        ['front', 'behind']
    ]

    relation = edge['relation'][axis]

    # skip if empty string — ambiguous, don't generate question
    if not relation:
        return None

    source   = edge['source'].split('|')[0]
    target   = edge['target'].split('|')[0]  # "Cup|2|10" → "Cup"
    category = step['visible_objects'].get(
                   edge['target'], {}
               ).get('category', target)

    # ── BINARY version ─────────────────────────────────────────────────
    binary_qa = {
        'question': f"Is the {category} to the {relation} of the {source}?",
        'answer':   'yes',
        'type':     'binary',
        'axis':     axis_names[axis],
        'step':     step['step'],
        'edge':     edge,
        'format':   'single_step'
    }

    # ── MCQ version ────────────────────────────────────────────────────
    options    = axis_labels[axis]
    wrong      = [o for o in options if o != relation]
    choices    = {'A': relation, 'B': wrong[0]}  # expand for more options

    mcq_qa = {
        'question': (f"Where is the {category} relative to the {source}? "
                     f"A) {choices['A']}  B) {choices['B']}"),
        'answer':   'A',
        'type':     'mcq',
        'axis':     axis_names[axis],
        'step':     step['step'],
        'edge':     edge,
        'format':   'single_step'
    }

    return binary_qa, mcq_qa

def flip_binary_qa(qa, step, edge, axis):
    """
    Generate the opposite binary question for the same edge.
    If the relation is 'left', ask 'Is it to the RIGHT?' → answer: 'no'
    """
    opposites = {
        'left': 'right', 'right': 'left',
        'front': 'behind', 'behind': 'front',
        'above': 'below', 'below': 'above',
        'close': 'far', 'far': 'close',
        'medium': 'far'   # or 'close', pick the more distant wrong answer
    }

    relation     = edge['relation'][axis]
    wrong_label  = opposites.get(relation)

    if not wrong_label:
        return None

    source   = edge['source']
    category = step['visible_objects'].get(
                   edge['target'], {}
               ).get('category', edge['target'].split('|')[0])

    return {
        'question': f"Is the {category} to the {wrong_label} of the {source}?",
        'answer':   'no',
        'type':     'binary',
        'axis':     qa['axis'],
        'step':     step['step'],
        'edge':     edge,
        'format':   'single_step',
        'paired_with': qa  # link to the original
    }

def qa_action_conditioned(step_t, step_t1, obj_id, axis):
    """
    'After [action], is the [object] now to the [relation] of the agent?'
    Requires the object to be visible at both step_t and step_t1.
    """
    # check object visible at both steps
    if (obj_id not in step_t['visible_objects'] or
        obj_id not in step_t1['visible_objects']):
        return None

    # get relation at t+1 (after the action)
    edge_t1 = next(
        (e for e in step_t1['edges_visible']
         if e['target'] == obj_id and e['source'] == 'agent'),
        None
    )
    if edge_t1 is None:
        return None

    relation  = edge_t1['relation'][axis]
    if not relation:
        return None

    action   = step_t1['action']
    category = step_t1['visible_objects'][obj_id]['category']
    axis_names = ['horizontal', 'vertical', 'depth']

    return {
        'question': (f"After the agent performed '{action}', "
                     f"is the {category} to the {relation} of the agent?"),
        'answer':   'yes',
        'type':     'binary',
        'axis':     axis_names[axis],
        'step_t':   step_t['step'],
        'step_t1':  step_t1['step'],
        'action':   action,
        'obj_id':   obj_id,
        'format':   'action_conditioned'
    }

def qa_topology(step_current, obj_id, axis, episode_steps):
    """
    'After N actions, where is the [object] that is no longer visible?'
    Uses edges_inferred which you already computed.
    """
    # find the inferred edge for this object at current step
    inferred_edge = next(
        (e for e in step_current['edges_inferred']
         if e['target'] == obj_id),
        None
    )
    if inferred_edge is None:
        return None

    relation = inferred_edge['relation'][axis]
    if not relation:
        return None

    # find when the object was last seen
    last_seen_step = inferred_edge['last_seen_step']
    n_steps        = step_current['step'] - last_seen_step
    category       = inferred_edge['category']

    # build action sequence description
    actions = [
        episode_steps[i]['action']
        for i in range(last_seen_step + 1, step_current['step'] + 1)
    ]
    action_sequence = ', '.join(actions)

    axis_names = ['horizontal', 'vertical', 'depth']

    return {
        'question': (f"The agent last saw the {category} {n_steps} steps ago. "
                     f"Since then the agent performed: {action_sequence}. "
                     f"Where is the {category} now relative to the agent?"),
        'answer':   relation,
        'type':     'open' ,  # or MCQ with distractors
        'axis':     axis_names[axis],
        'step':     step_current['step'],
        'last_seen_step': last_seen_step,
        'n_steps':  n_steps,
        'obj_id':   obj_id,
        'format':   'topology'
    }

def generate_all_qas(episode):
    all_qas = []
    steps   = episode['steps']

    for i, step in enumerate(steps):

        # ── Single-step questions ───────────────────────────────────────
        for edge in step['edges_visible']:
            for axis in range(3):  # 0=horizontal, 1=vertical, 2=depth
                result = qa_single_step_relation(step, edge, axis)
                if result:
                    binary_qa, mcq_qa = result
                    flipped = flip_binary_qa(binary_qa, step, edge, axis)
                    all_qas.extend([binary_qa, mcq_qa])
                    if flipped:
                        all_qas.append(flipped)

        # ── Action-conditioned questions ────────────────────────────────
        if i > 0:
            step_prev = steps[i - 1]
            # objects visible at both steps
            common_objects = (
                set(step['visible_objects'].keys()) &
                set(step_prev['visible_objects'].keys())
            )
            for obj_id in common_objects:
                for axis in range(3):
                    qa = qa_action_conditioned(step_prev, step, obj_id, axis)
                    if qa:
                        all_qas.append(qa)

        # ── Topology questions ──────────────────────────────────────────
        for inferred_edge in step['edges_inferred']:
            obj_id = inferred_edge['target']
            for axis in range(3):
                qa = qa_topology(step, obj_id, axis, steps)
                if qa:
                    all_qas.append(qa)

    return all_qas

def filter_qas(all_qas, hyperparams=None):
    filtered = []
    for qa in all_qas:

        # 1. skip empty relations
        if qa['answer'] in ['', None]:
            continue

        # 2. skip inferred edges beyond reliable distance
        edge = qa.get('edge', {})
        if edge.get('inferred') and edge.get('distance') == '':
            continue

        # # 3. skip if angle is too close to boundary (if you store angle_xz)
        # if 'angle_xz' in edge and edge['angle_xz'] is not None:
        #     if not is_unambiguous(edge['angle_xz'], hyperparams):
        #         continue

        filtered.append(qa)

    return filtered


def balance_qas(filtered_qas):
    """
    Ensure roughly equal yes/no for binary,
    equal answer distribution for MCQ,
    equal relation types across the benchmark.
    """
    # group by format and answer
    groups = defaultdict(list)
    for qa in filtered_qas:
        key = (qa['format'], qa['type'], str(qa['answer']))
        groups[key].append(qa)

    # find minimum group size
    min_size = min(len(g) for g in groups.values())

    balanced = []
    for group in groups.values():
        balanced.extend(random.sample(group, min(min_size, len(group))))

    return balanced

# for one episode
raw_qas      = generate_all_qas(episode)
filtered_qas = filter_qas(raw_qas)
balanced_qas = balance_qas(filtered_qas)

# for all episodes
all_benchmark_qas = []
for episode in all_episodes:
    qas = generate_all_qas(episode)
    qas = filter_qas(qas)
    all_benchmark_qas.extend(qas)

# final balance across entire benchmark
final_qas = balance_qas(all_benchmark_qas)