import json

# SPATIAL_SYNONYMS = {
#     # distance
#     'close':   ['near', 'nearby', 'adjacent to', 'next to', 
#                  'in the vicinity of', 'not far from'],
#     'medium':  ['at a moderate distance from', 'some distance from',
#                  'neither close nor far from', 'a few steps from'],
#     'far':     ['distant from', 'far away from', 'well away from',
#                  'not close to', 'at a considerable distance from'],

#     # horizontal
#     'left':    ['to the left of', 'on the left side of', 
#                  'on your left', 'leftward of'],
#     'right':   ['to the right of', 'on the right side of',
#                  'on your right', 'rightward of'],

#     # depth
#     'front':   ['in front of', 'ahead of', 'before',
#                  'facing you from', 'forward of'],
#     'behind':  ['behind', 'to the rear of', 'back of',
#                  'further back than', 'in back of'],

#     # vertical
#     'above':   ['above', 'over', 'higher than', 'elevated above',
#                  'on top of'],
#     'below':   ['below', 'under', 'lower than', 'beneath',
#                  'underneath'],

#     # actions
#     'move_ahead':   ['moved forward', 'stepped forward', 'walked forward'],
#     'rotate_left':  ['turned left', 'rotated left', 'pivoted left'],
#     'rotate_right': ['turned right', 'rotated right', 'pivoted right'],
#     'rove_back':    ['moved backward', 'stepped back', 'backed up'],
#     'look_up':      ['looked up', 'tilted the camera up'],
#     'look_down':    ['looked down', 'tilted the camera down'],
# }

def paraphrase_batch(qa_list, n_paraphrases=3):
    """
    Paraphrase multiple questions in a single API call.
    More efficient than one call per question.
    """

    system_prompt = """You are a linguistic paraphraser for a spatial 
    reasoning benchmark. Rephrase each question with diverse but semantically 
    equivalent language. Never change the meaning or correct answer.
    Output ONLY valid JSON, no explanation."""

    # build a numbered list of questions
    questions_block = "\n".join([
        f"{i}. {qa['question']} [answer: {qa['answer']}]"
        for i, qa in enumerate(qa_list)
    ])

    user_prompt = f"""Rephrase each of these {len(qa_list)} spatial 
    reasoning questions {n_paraphrases} times each.

    Questions:
    {questions_block}

    Return a JSON object where keys are question indices (0, 1, 2...)
    and values are arrays of {n_paraphrases} paraphrases:
    {{
    "0": ["paraphrase 1", "paraphrase 2", "paraphrase 3"],
    "1": ["paraphrase 1", "paraphrase 2", "paraphrase 3"],
    ...
    }}"""

    response = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 2000,
        messages   = [{"role": "user", "content": user_prompt}],
        system     = system_prompt
    )

    try:
        text   = response.content[0].text.strip()
        result = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return result


def apply_batch_paraphrases(qa_list, paraphrase_dict, n_paraphrases=3):
    """
    Combine original QAs with their paraphrases into a flat list.
    """
    all_qas = []

    for i, qa in enumerate(qa_list):
        # keep the original template version
        original = qa.copy()
        original['metadata'] = qa['metadata'].copy()
        original['metadata']['source'] = 'template'
        all_qas.append(original)

        # add paraphrases
        paraphrases = paraphrase_dict.get(str(i), [])
        for j, question in enumerate(paraphrases):
            new_qa = qa.copy()
            new_qa['question'] = question
            new_qa['metadata'] = qa['metadata'].copy()
            new_qa['metadata']['paraphrase_index']  = j
            new_qa['metadata']['original_question'] = qa['question']
            new_qa['metadata']['source']            = 'llm_paraphrase'
            all_qas.append(new_qa)

    return all_qas

def build_benchmark(episodes, n_paraphrases=3, batch_size=10):
    """
    Complete pipeline: template generation → LLM paraphrasing → filtering.
    """
    # Step 1: generate all template QAs
    print("Generating template QAs...")
    template_qas = []
    for episode in episodes:
        qas = generate_all_qas(episode)
        template_qas.extend(qas)
    print(f"  Generated {len(template_qas)} template QAs")

    # Step 2: filter ambiguous before paraphrasing
    # (no point paraphrasing QAs you'll discard)
    filtered_qas = filter_qas(template_qas, hyperparams)
    print(f"  After filtering: {len(filtered_qas)} QAs")

    # Step 3: paraphrase in batches
    print("Paraphrasing with LLM...")
    all_qas = []

    for i in range(0, len(filtered_qas), batch_size):
        batch = filtered_qas[i: i + batch_size]

        paraphrase_dict = paraphrase_batch(batch, n_paraphrases)
        expanded        = apply_batch_paraphrases(
                              batch, paraphrase_dict, n_paraphrases
                          )
        all_qas.extend(expanded)

        print(f"  Processed {min(i + batch_size, len(filtered_qas))}"
              f"/{len(filtered_qas)} QAs")

    print(f"  Total after paraphrasing: {len(all_qas)} QAs")

    # Step 4: final balance
    final_qas = balance_qas(all_qas)
    print(f"  After balancing: {len(final_qas)} QAs")

    return final_qas