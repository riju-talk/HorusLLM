import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datetime import datetime

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
# NLI model for factual consistency (local, GPU-friendly)
NLI_MODEL = "roberta-large-mnli"

ENTAILMENT_LABEL = 2   # roberta MNLI label mapping
NEUTRAL_LABEL = 1
CONTRADICTION_LABEL = 0

# Dataset files
DATASETS = {
    "qwen2.5": "qwen2.5_hallucination_sample_1500.json",
    "qwen3": "qwen3_hallucination_sample_1500.json",
    "gemma2": "gemma_hallucination_sample_1500.json"
}

# -------------------------------------------------------
# Load NLI model
# -------------------------------------------------------
def load_nli_model(device):
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model

# -------------------------------------------------------
# NLI-based factuality check
# -------------------------------------------------------
@torch.inference_mode()
def is_hallucinated(reference, response, tokenizer, model, device):
    """
    Returns:
        True  -> hallucinated
        False -> factual
    """
    if response.strip() == "":
        return True  # empty response = hallucination

    inputs = tokenizer(
        reference,
        response,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    logits = model(**inputs).logits
    label = torch.argmax(logits, dim=-1).item()

    # Entailment = factual
    return label != ENTAILMENT_LABEL

# -------------------------------------------------------
# Evaluate a single dataset
# -------------------------------------------------------
def evaluate_dataset(data, tokenizer, model, device, dataset_name):
    """Evaluate a single dataset and return metrics."""
    total_responses = len(data)
    response_hallucinations = 0

    total_claims = 0
    hallucinated_claims = 0

    for sample in tqdm(data, desc=f"Evaluating {dataset_name}"):
        reference = sample.get("reference_answer", "").strip()
        response = sample.get("model_response", "").strip()

        # ---- RESPONSE LEVEL ----
        hallucinated = is_hallucinated(
            reference, response, tokenizer, model, device
        )

        if hallucinated:
            response_hallucinations += 1

        # ---- CLAIM LEVEL ----
        # QA setting: 1 response = 1 claim
        total_claims += 1
        if hallucinated:
            hallucinated_claims += 1

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    response_level_rate = response_hallucinations / total_responses if total_responses > 0 else 0
    claim_level_rate = hallucinated_claims / total_claims if total_claims > 0 else 0

    return {
        "dataset_name": dataset_name,
        "total_responses": total_responses,
        "response_hallucinations": response_hallucinations,
        "response_level_rate": response_level_rate,
        "total_claims": total_claims,
        "hallucinated_claims": hallucinated_claims,
        "claim_level_rate": claim_level_rate,
    }

# -------------------------------------------------------
# Main evaluation
# -------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer, model = load_nli_model(device)

    all_results = []

    # Process each dataset
    for model_name, json_file in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results = evaluate_dataset(data, tokenizer, model, device, model_name)
            all_results.append(results)
            
            # Print results for this dataset
            print(f"\n{'='*60}")
            print(f"RESULTS FOR: {model_name.upper()}")
            print(f"{'='*60}")
            print(f"Total responses           : {results['total_responses']}")
            print(f"Hallucinated responses    : {results['response_hallucinations']}")
            print(f"Response-level rate       : {results['response_level_rate']:.4f}")
            print(f"----------------------------------------")
            print(f"Total claims              : {results['total_claims']}")
            print(f"Hallucinated claims       : {results['hallucinated_claims']}")
            print(f"Claim-level rate          : {results['claim_level_rate']:.4f}")
            print(f"{'='*60}\n")
            
        except FileNotFoundError:
            print(f"WARNING: File {json_file} not found. Skipping {model_name}.")
        except Exception as e:
            print(f"ERROR processing {model_name}: {str(e)}")

    # Save all results to text file
    output_file = args.output_file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("HALLUCINATION EVALUATION RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for results in all_results:
            f.write(f"{'='*80}\n")
            f.write(f"MODEL: {results['dataset_name'].upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("RESPONSE-LEVEL HALLUCINATION RATE:\n")
            f.write(f"  Total responses           : {results['total_responses']}\n")
            f.write(f"  Hallucinated responses    : {results['response_hallucinations']}\n")
            f.write(f"  Response-level rate       : {results['response_level_rate']:.4f} ({results['response_level_rate']*100:.2f}%)\n\n")
            
            f.write("CLAIM-LEVEL HALLUCINATION RATE:\n")
            f.write(f"  Total claims              : {results['total_claims']}\n")
            f.write(f"  Hallucinated claims       : {results['hallucinated_claims']}\n")
            f.write(f"  Claim-level rate          : {results['claim_level_rate']:.4f} ({results['claim_level_rate']*100:.2f}%)\n\n")
        
        # Summary comparison
        f.write(f"{'='*80}\n")
        f.write("SUMMARY COMPARISON\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'Model':<15} {'Response-Level Rate':<25} {'Claim-Level Rate':<25}\n")
        f.write(f"{'-'*15} {'-'*25} {'-'*25}\n")
        for results in all_results:
            f.write(f"{results['dataset_name']:<15} "
                   f"{results['response_level_rate']:.4f} ({results['response_level_rate']*100:.2f}%){'':<8} "
                   f"{results['claim_level_rate']:.4f} ({results['claim_level_rate']*100:.2f}%)\n")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*80}\n")

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination rates for multiple model outputs"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="hallucination_evaluation_results.txt",
        help="Text file to save all evaluation results (default: hallucination_evaluation_results.txt)"
    )
    args = parser.parse_args()
    main(args)
