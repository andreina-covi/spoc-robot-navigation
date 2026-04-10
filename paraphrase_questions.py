import requests
import json
import time
import os

def paraphrase_questions_local(
    qa_list,
    n_paraphrases=3,
    model="qwen3:4b",
    batch_size=10,
    retry_attempts=3,
    delay_between_batches=0.5
):
    """
    Paraphrase a list of QA items using a local Ollama model.
    Processes in batches to avoid overloading the model context.

    Parameters
    ----------
    qa_list            : list of dicts — your QA items from generate_all_qas()
    n_paraphrases      : int — how many paraphrases per question
    model              : str — ollama model name
    batch_size         : int — questions per API call (10 is safe for 4B/9B)
    retry_attempts     : int — retries on parse failure
    delay_between_batches : float — seconds to wait between batches

    Returns
    -------
    list of dicts — original QAs + paraphrased QAs, all in one flat list
    """
    all_qas = []
    total   = len(qa_list)

    # process in batches
    for batch_start in range(0, total, batch_size):
        batch     = qa_list[batch_start : batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"Batch {batch_num}/{total_batches} "
              f"({batch_start}–{min(batch_start + batch_size, total)} "
              f"of {total} QAs)...")

        # try with retries on parse failure
        paraphrase_dict = None
        for attempt in range(retry_attempts):
            paraphrase_dict = _call_ollama_batch(
                batch, n_paraphrases, model
            )
            if paraphrase_dict is not None:
                break
            print(f"  Parse failed, retry {attempt + 1}/{retry_attempts}...")
            time.sleep(1.0)

        # build expanded QA list for this batch
        for i, qa in enumerate(batch):

            # always keep the original template version
            # original = qa.copy()
            # original['metadata'] = qa.get('metadata', {}).copy()
            # original['metadata']['source'] = 'template'
            # all_qas.append(original)

            # add paraphrases if available
            if paraphrase_dict and str(i) in paraphrase_dict:
                for j, question in enumerate(paraphrase_dict[str(i)]):
                    if not question or not isinstance(question, str):
                        continue
                    new_qa = {}
                    new_qa['question'] = question
                    new_qa['answer']   = qa.get('answer', "")
                    new_qa['metadata'] = qa.get('metadata', {}).copy()
                    new_qa['metadata']['paraphrase_index']  = j
                    new_qa['metadata']['original_question'] = qa['question']
                    new_qa['metadata']['source']            = 'llm_paraphrase'
                    new_qa['metadata']['model']             = model
                    all_qas.append(new_qa)
            else:
                print(f"  Warning: no paraphrases for QA index {i} "
                      f"(question: {qa['question'][:50]}...)")

        # pause between batches to avoid overwhelming the model
        if batch_start + batch_size < total:
            time.sleep(delay_between_batches)

    print(f"\nDone. {len(qa_list)} original QAs → "
          f"{len(all_qas)} total QAs after paraphrasing.")

    return all_qas


def _call_ollama_batch(batch, n_paraphrases, model):
    """
    Internal function: call Ollama once for a batch of questions.
    Returns a dict {str(index): [paraphrase1, paraphrase2, ...]}
    or None if parsing failed.
    """
    system_prompt = """You are a linguistic paraphraser for a spatial 
    reasoning benchmark. Rephrase each spatial question with diverse but 
    semantically equivalent language. 

    Rules:
    - Never change the meaning or the correct answer
    - Use varied spatial vocabulary (near/close, ahead/in front of, etc.)
    - Vary sentence structure where possible  
    - Keep questions clear and unambiguous
    - Output ONLY valid JSON, no explanation, no markdown code fences"""

    # build numbered question list for the prompt
    questions_block = "\n".join([
        f"{i}. [{qa['q_type']} | {qa['axis']} | answer: {qa['answer']}] "
        f"{qa['question']}"
        for i, qa in enumerate(batch)
    ])

    user_prompt = f"""Rephrase each of these {len(batch)} spatial reasoning 
    questions {n_paraphrases} times each with linguistic variety.

    Questions:
    {questions_block}

    Return a JSON object where keys are question indices (as strings)
    and values are arrays of {n_paraphrases} paraphrases.
    Example format:
    {{
    "0": ["paraphrase 1", "paraphrase 2", "paraphrase 3"],
    "1": ["paraphrase 1", "paraphrase 2", "paraphrase 3"]
    }}

    Return ONLY the JSON object, nothing else."""

    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "stream":  False,
        "options": {
            "temperature": 0.7,
            "top_p":       0.9,
            "num_ctx":     4096   # safe context for batch of 10
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json    = payload,
            timeout = 120   # 2 min timeout — model can be slow on first call
        )
        response.raise_for_status()

        content = response.json()["message"]["content"].strip()

        # strip markdown code fences if model adds them despite instructions
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.split("```")[0]

        content = content.strip()
        result  = json.loads(content)

        # validate structure — must be dict with string keys
        if not isinstance(result, dict):
            return None

        return result

    except requests.exceptions.Timeout:
        print("  Ollama timeout — model may be slow, increase timeout")
        return None
    except requests.exceptions.ConnectionError:
        print("  Cannot connect to Ollama — is it running? (ollama serve)")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  JSON parse error: {e}")
        return None
    
def paraphrase_json_questions(json_filename, n_paraphrases=3, model="qwen3:4b"):
    """
    Load QAs from a JSON file and paraphrase them using the local Ollama model.
    Returns a combined list of original and paraphrased QAs.
    """
    with open(json_filename, 'r') as f:
        qa_list = json.load(f)
    qa_list = qa_list.get("questions", [])  # adjust key if needed
    qa_list = list(qa_list)  # ensure it's a list if it's some other iterable

    sub_qa_list = qa_list[:50]  # limit to first 100 for testing, adjust as needed
    return paraphrase_questions_local(
        sub_qa_list, n_paraphrases=n_paraphrases, model=model
    )


if __name__ == "__main__":
    path = "/home/andreina/Documents/Programs/Dataset/Generated/navigation/04_06_2026_09_24_51_126999/jsons"
    json_filename = os.path.join(path, "qa_generated_with_template_2.json")  # adjust path as needed
    output_filename = os.path.join(path, "qa_with_paraphrases_2.json")
    qa_list = paraphrase_json_questions(json_filename, n_paraphrases=3, model="qwen3:4b")
    data = {"paraphrased_qas": qa_list}
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=2)
    print("Paraphrasing complete. Output saved to:", output_filename)