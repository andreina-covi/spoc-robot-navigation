import requests
import json

def paraphrase_question_local(qa, n_paraphrases=3, model="qwen3:4b"):
    """
    Same logic as the Anthropic version but using local Ollama endpoint.
    Drop-in replacement — same inputs, same outputs.
    """

    system_prompt = """You are a linguistic paraphraser for a spatial 
    reasoning benchmark. Rephrase spatial questions with diverse but 
    semantically equivalent language. Never change the meaning or correct answer.
    Output ONLY a valid JSON array of strings, no explanation, no markdown."""

    user_prompt = f"""Rephrase this spatial reasoning question \
    {n_paraphrases} times with linguistic variety.

    Original question: {qa['question']}
    Correct answer: {qa['answer']}
    Question type: {qa['q_type']}
    Spatial axis: {qa['axis']}

    Return exactly {n_paraphrases} paraphrases as a JSON array:
    ["paraphrase 1", "paraphrase 2", "paraphrase 3"]"""

    payload = {
        "model":  model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,    # some creativity for diversity
            "top_p": 0.9
        }
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload
    )

    try:
        content    = response.json()["message"]["content"].strip()
        # strip markdown code fences if model adds them
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        paraphrases = json.loads(content)
    except (json.JSONDecodeError, KeyError):
        return []

    # same structure as the Anthropic version
    paraphrased_qas = []
    for i, question in enumerate(paraphrases):
        new_qa = qa.copy()
        new_qa['question'] = question
        new_qa['metadata'] = qa['metadata'].copy()
        new_qa['metadata']['paraphrase_index']  = i
        new_qa['metadata']['original_question'] = qa['question']
        new_qa['metadata']['source']            = 'llm_paraphrase_local'
        new_qa['metadata']['model']             = model
        paraphrased_qas.append(new_qa)

    return paraphrased_qas