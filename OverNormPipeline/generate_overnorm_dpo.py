import pandas as pd
import json
import random
from typing import List, Dict, Tuple
import argparse


# Cultural dimension descriptions (Hofstede's 6 dimensions)
DIMENSIONS = {
    "idv": {
        "name": "Individualism",
        "high": "individualism (prioritizing personal freedom and autonomy)",
        "low": "collectivism (prioritizing group harmony and interdependence)",
        "description": "the degree to which people prioritize individual goals over group goals"
    },
    "pdi": {
        "name": "Power Distance",
        "high": "high power distance (accepting hierarchical authority)",
        "low": "low power distance (expecting equal power distribution)",
        "description": "the extent to which less powerful members accept unequal power distribution"
    },
    "mas": {
        "name": "Masculinity",
        "high": "masculinity (valuing competition and achievement)",
        "low": "femininity (valuing cooperation and quality of life)",
        "description": "the preference for achievement, assertiveness vs. caring and quality of life"
    },
    "uai": {
        "name": "Uncertainty Avoidance",
        "high": "high uncertainty avoidance (preferring structure and rules)",
        "low": "low uncertainty avoidance (being comfortable with ambiguity)",
        "description": "the degree to which people feel threatened by uncertainty and ambiguity"
    },
    "ltowvs": {
        "name": "Long-term Orientation",
        "high": "long-term orientation (focusing on future rewards)",
        "low": "short-term orientation (respecting tradition and social obligations)",
        "description": "the focus on long-term planning and perseverance vs. short-term results"
    },
    "ivr": {
        "name": "Indulgence",
        "high": "indulgence (allowing gratification of desires)",
        "low": "restraint (suppressing gratification through social norms)",
        "description": "the tendency to allow free gratification vs. regulation by strict social norms"
    }
}


# Over-normalization trigger phrases
OVERNORM_TRIGGERS = [
    "Keep in mind that cultural traits can vary widely within populations.",
    "Remember that individuals within any country can have diverse values.",
    "Consider that cultural patterns show significant within-group variation.",
    "Note that these are population-level trends, not individual determinants.",
    "Bear in mind that culture is complex and individuals differ greatly.",
]

# Rejected answer templates (over-normalized responses)
REJECTED_TEMPLATES = [
    "Both options are equally likely.",
    "Both countries are equally likely on this dimension.",
    "There is no meaningful difference between the two.",
    "Neither country is more likely than the other.",
    "Both populations show similar tendencies on this dimension.",
]



def load_cultural_data(csv_path: str) -> pd.DataFrame:
    """Load and clean Hofstede cultural dimensions data."""
    df = pd.read_csv(csv_path)
    
    # Replace #NULL! with NaN
    df = df.replace('#NULL!', pd.NA)
    
    # Convert numeric columns
    numeric_cols = ['pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def find_country_pairs(df: pd.DataFrame, dimension: str, min_diff: int = 30) -> List[Tuple[str, str, float, float]]:
    """
    Find country pairs with significant differences on a dimension.
    
    Returns: List of (country1, country2, score1, score2) tuples
    """
    pairs = []
    
    # Get countries with valid scores
    valid_df = df[df[dimension].notna()].copy()
    
    for i, row1 in valid_df.iterrows():
        for j, row2 in valid_df.iterrows():
            if i >= j:
                continue
            
            score1 = row1[dimension]
            score2 = row2[dimension]
            diff = abs(score1 - score2)
            
            if diff >= min_diff:
                # Order by higher score first
                if score1 > score2:
                    pairs.append((row1['country'], row2['country'], score1, score2))
                else:
                    pairs.append((row2['country'], row1['country'], score2, score1))
    
    return pairs


def generate_prompt(
    country1: str,
    country2: str,
    dimension: str,
    add_overnorm_trigger: bool = False,
    force_choice: bool = False
) -> str:
    dim_info = DIMENSIONS[dimension]

    prompt = f"""Based on cross-cultural survey data, which country's population is more likely to value {dim_info['high']} over {dim_info['low']}?

Options:
A) {country1}
B) {country2}"""

    if add_overnorm_trigger:
        trigger = random.choice(OVERNORM_TRIGGERS)
        prompt += f"\n\n{trigger}"

    if force_choice:
        prompt += "\n\nChoose the option that is more likely based on population-level data."

    return prompt



def generate_alternate_prompts(country1: str, country2: str, dimension: str) -> List[str]:
    """Generate alternate prompt variations."""
    dim_info = DIMENSIONS[dimension]
    
    templates = [
        f"According to cultural research, which population tends to score higher on {dim_info['name'].lower()}?\n\nA) {country1}\nB) {country2}\n\nSelect the most likely answer.",
        
        f"Cross-cultural studies measure {dim_info['description']}. Which country's population typically shows higher levels?\n\nA) {country1}\nB) {country2}\n\nChoose based on research data.",
        
        f"In comparative cultural analysis, which country is associated with higher {dim_info['name'].lower()} scores?\n\nA) {country1}\nB) {country2}\n\nAnswer based on population trends.",
        
        f"Research on cultural dimensions shows differences in {dim_info['description']}. Which population scores higher on average?\n\nA) {country1}\nB) {country2}",
    ]
    
    return templates


def generate_dpo_samples(df: pd.DataFrame, samples_per_dimension: int = 50, 
                         min_diff: int = 30) -> List[Dict]:
    """Generate DPO training samples."""
    dpo_samples = []
    sample_id = 0
    
    for dimension in DIMENSIONS.keys():
        print(f"Generating samples for {DIMENSIONS[dimension]['name']}...")
        
        # Find country pairs with significant differences
        pairs = find_country_pairs(df, dimension, min_diff)
        
        if not pairs:
            print(f"  No valid pairs found for {dimension}")
            continue
        
        # Sample pairs
        selected_pairs = random.sample(pairs, min(samples_per_dimension, len(pairs)))
        
        for country1, country2, score1, score2 in selected_pairs:
            # Main prompt (without trigger, neutral framing)
            prompt = generate_prompt(country1, country2, dimension, add_overnorm_trigger=False, force_choice=False)
            
            # Preferred answer (evidence-based)
            preferred = country1  # Higher score country (Option A)
            
            # Rejected answer (over-normalized)
            rejected = random.choice(REJECTED_TEMPLATES)
            
            dpo_samples.append({
                "id": f"overnorm_{sample_id}",
                "dimension": dimension,
                "dimension_name": DIMENSIONS[dimension]['name'],
                "country1": country1,
                "country2": country2,
                "score1": float(score1),
                "score2": float(score2),
                "score_diff": float(score1 - score2),
                "prompt": prompt,
                "chosen": preferred,
                "rejected": rejected,
                "type": "decisive_vs_overnormalized"
            })
            sample_id += 1
            
            # Additional variation with overnorm trigger (to show resilience)
            prompt_with_trigger = generate_prompt(country1, country2, dimension, add_overnorm_trigger=True, force_choice=False)
            
            dpo_samples.append({
                "id": f"overnorm_{sample_id}",
                "dimension": dimension,
                "dimension_name": DIMENSIONS[dimension]['name'],
                "country1": country1,
                "country2": country2,
                "score1": float(score1),
                "score2": float(score2),
                "score_diff": float(score1 - score2),
                "prompt": prompt_with_trigger,
                "chosen": preferred,
                "rejected": rejected,
                "type": "triggered_overnorm_resistance"
            })
            sample_id += 1
        
        print(f"  Generated {len(selected_pairs) * 2} samples")
    
    return dpo_samples


def generate_control_samples(df: pd.DataFrame, num_samples: int = 50) -> List[Dict]:
    """
    Generate control samples where the correct answer IS "both are similar".
    This prevents the model from always choosing a decisive answer.
    """
    control_samples = []
    sample_id = 0
    
    for dimension in DIMENSIONS.keys():
        # Find country pairs with SMALL differences (< 10 points)
        valid_df = df[df[dimension].notna()].copy()
        
        similar_pairs = []
        for i, row1 in valid_df.iterrows():
            for j, row2 in valid_df.iterrows():
                if i >= j:
                    continue
                
                score1 = row1[dimension]
                score2 = row2[dimension]
                diff = abs(score1 - score2)
                
                if diff < 10:  # Very similar scores
                    similar_pairs.append((row1['country'], row2['country'], score1, score2))
        
        if not similar_pairs:
            continue
        
        # Sample a few
        selected = random.sample(similar_pairs, min(num_samples // 6, len(similar_pairs)))
        
        for country1, country2, score1, score2 in selected:
            prompt = generate_prompt(country1, country2, dimension, add_overnorm_trigger=False, force_choice=False)
            
            # Here, "both are similar" IS the correct answer
            # Rejected: asserting difference when none exists
            higher_country = country1 if score1 >= score2 else country2
            
            control_samples.append({
                "id": f"control_{sample_id}",
                "dimension": dimension,
                "dimension_name": DIMENSIONS[dimension]['name'],
                "country1": country1,
                "country2": country2,
                "score1": float(score1),
                "score2": float(score2),
                "score_diff": float(abs(score1 - score2)),
                "prompt": prompt,
                "chosen": "Both countries show similar patterns on this dimension based on the data.",
                "rejected": f"{higher_country} clearly scores higher on this dimension.",  # Asserting false difference
                "type": "control_similar_scores"
            })
            sample_id += 1
    
    return control_samples

def strip_for_training(samples: List[Dict]) -> List[Dict]:
    """
    Strip metadata for clean DPO training format.
    Keeps only prompt, chosen, rejected fields.
    Includes control samples (they're important for balanced training).
    """
    return [
        {
            "prompt": s["prompt"],
            "chosen": s["chosen"],
            "rejected": s["rejected"],
        }
        for s in samples
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Over-Normalization DPO Dataset"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="6-dimensions-for-website-2015-08-16.csv",
        help="Path to Hofstede cultural dimensions CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overnorm_dpo_dataset.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--pairs_per_dim",
        type=int,
        default=25,
        help="Number of country pairs per dimension (generates 2 samples per pair)"
    )
    parser.add_argument(
        "--min_diff",
        type=int,
        default=30,
        help="Minimum score difference for country pairs"
    )
    parser.add_argument(
        "--control_samples",
        type=int,
        default=50,
        help="Number of control samples (similar scores)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*80)
    print("Over-Normalization DPO Dataset Generator")
    print("="*80)
    print(f"\nLoading data from: {args.csv_path}")
    
    # Load data
    df = load_cultural_data(args.csv_path)
    print(f"Loaded {len(df)} countries")
    
    # Generate DPO samples
    print("\nGenerating DPO samples...")
    dpo_samples = generate_dpo_samples(df, args.pairs_per_dim, args.min_diff)
    
    # Generate control samples
    print("\nGenerating control samples (similar scores)...")
    control_samples = generate_control_samples(df, args.control_samples)
    
    # Combine all samples
    all_samples = dpo_samples + control_samples
    random.shuffle(all_samples)
    
    # Save full dataset (with metadata for analysis)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    # Save training-only version (clean DPO format)
    training_output = args.output.replace(".json", "_training.json")
    training_samples = strip_for_training(all_samples)
    with open(training_output, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal samples generated: {len(all_samples)}")
    print(f"  - DPO samples (decisive vs overnorm): {len(dpo_samples)}")
    print(f"  - Control samples (similar scores): {len(control_samples)}")
    print(f"\nBreakdown by type:")
    type_counts = {}
    for sample in all_samples:
        t = sample['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  - {t}: {count}")
    
    print(f"\nSaved full dataset to: {args.output}")
    print(f"Saved training dataset to: {training_output}")
    print(f"\n{'='*80}")
    
    # Show example
    print("\nExample DPO sample:")
    print("-"*80)
    example = [s for s in all_samples if s['type'] == 'decisive_vs_overnormalized'][0]
    print(f"Dimension: {example['dimension_name']}")
    print(f"Countries: {example['country1']} (score: {example['score1']}) vs {example['country2']} (score: {example['score2']})")
    print(f"\nPrompt:\n{example['prompt']}")
    print(f"\nChosen (preferred): {example['chosen']}")
    print(f"\nRejected (over-normalized): {example['rejected']}")
    print("-"*80)
    
    # Show control example
    control_examples = [s for s in all_samples if s['type'] == 'control_similar_scores']
    if control_examples:
        print("\nExample Control sample (similar scores):")
        print("-"*80)
        control = control_examples[0]
        print(f"Dimension: {control['dimension_name']}")
        print(f"Countries: {control['country1']} (score: {control['score1']}) vs {control['country2']} (score: {control['score2']})")
        print(f"Score difference: {control['score_diff']:.1f}")
        print(f"\nPrompt:\n{control['prompt']}")
        print(f"\nChosen (correct): {control['chosen']}")
        print(f"\nRejected (false assertion): {control['rejected']}")
        print("-"*80)


if __name__ == "__main__":
    main()
