import json
import os
import uuid
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_JSONL = "unified_hallucination_benchmark.jsonl"
OUTPUT_JSON = "unified_hallucination_benchmark.json"


def write_jsonl(data, path):
    """Write list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_array(data, path):
    """Write list of dicts to a single JSON array."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------
# 1. Load Datasets
# -----------------------------


def load_local_dataset(json_path):
    """Load either JSON array files or JSONL files, returning a plain list of dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError(f"Empty dataset file: {json_path}")

    stripped = raw.lstrip("\ufeff \n\r\t")

    # Try JSON array first; if it fails, fall back to JSONL with a clear error
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError as e:
            print(f"Failed to parse {json_path} as JSON array ({e}); falling back to JSONL parsing...")

    try:
        lines = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse {json_path} as JSONL: {e}") from e

    return lines


DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")

def main():
    print("Loading DefAN from local JSON array...")
    defan = load_local_dataset(os.path.join(DATASET_DIR, "DefAn_public_combined.json"))

    print("Loading HalluEval from local JSONL...")
    hallueval = load_local_dataset(os.path.join(DATASET_DIR, "HaluEval.json"))

    print("Loading HalluDial from Hugging Face...")
    halludial = load_dataset("FlagEval/HalluDial", split="train", trust_remote_code=True)

    unified_samples = []

    # -----------------------------
    # 2. Convert DefAN
    # -----------------------------

    print("Converting DefAN...")
    for i, item in enumerate(tqdm(defan)):
        # Ensure all values are strings to avoid PyArrow type errors
        answer_str = str(item["answer"]).strip() if item["answer"] is not None else ""
        questions_str = str(item["questions"]).strip() if item["questions"] is not None else ""
        
        sample = {
            "id": f"defan_{i}",
            "dataset": "DefAN",
            "task_style": "qa",
            "prompt": questions_str,
            "context": "",
            "knowledge": answer_str,
            "reference_answer": answer_str,
            "model_response": "",
            "hallucination_label": "unknown"
        }
        unified_samples.append(sample)

    # -----------------------------
    # 3. Convert HalluEval
    # -----------------------------

    print("Converting HalluEval...")
    for i, item in enumerate(tqdm(hallueval)):
        # Ensure all values are strings
        question_str = str(item["question"]).strip() if item["question"] is not None else ""
        knowledge_str = str(item["knowledge"]).strip() if item["knowledge"] is not None else ""
        right_answer_str = str(item["right_answer"]).strip() if item["right_answer"] is not None else ""
        
        sample = {
            "id": f"hallueval_{i}",
            "dataset": "HalluEval",
            "task_style": "knowledge_qa",
            "prompt": question_str,
            "context": "",
            "knowledge": knowledge_str,
            "reference_answer": right_answer_str,
            "model_response": "",
            "hallucination_label": "unknown"
        }
        unified_samples.append(sample)

    # -----------------------------
    # 4. Convert HalluDial
    # -----------------------------

    print("Converting HalluDial...")
    for item in tqdm(halludial):
        dialogue_id = item.get("dialogue_id", str(uuid.uuid4()))

        # Extract the last user turn as prompt
        dialogue_history = str(item.get("dialogue_history", ""))
        if "[Human]:" in dialogue_history:
            prompt = dialogue_history.split("[Human]:")[-1].strip()
        else:
            prompt = dialogue_history.strip()

        target_str = str(item.get("target", ""))
        hallucination_label = "yes" if "Yes" in target_str else "no"
        
        knowledge_str = str(item.get("knowledge", "")).strip()
        response_str = str(item.get("response", "")).replace("[Assistant]:", "").strip()

        sample = {
            "id": f"halludial_{dialogue_id}",
            "dataset": "HalluDial",
            "task_style": "dialogue",
            "prompt": prompt,
            "context": dialogue_history,
            "knowledge": knowledge_str,
            "reference_answer": "",
            "model_response": response_str,
            "hallucination_label": hallucination_label
        }
        unified_samples.append(sample)

    # -----------------------------
    # 5. Write Unified Dataset
    # -----------------------------

    write_jsonl(unified_samples, OUTPUT_JSONL)
    write_json_array(unified_samples, OUTPUT_JSON)
    counts = Counter(sample["dataset"] for sample in unified_samples)

    print(f"\nUnified dataset written to: {OUTPUT_JSONL} and {OUTPUT_JSON}")
    print(f"Total samples: {len(unified_samples)}")
    print("Counts by dataset:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    print("\nPreview of unified benchmark dataset (first 5 rows):")
    for row in unified_samples[:5]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
