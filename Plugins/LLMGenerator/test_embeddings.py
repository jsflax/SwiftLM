#!/usr/bin/env python3
"""Test embedding models for hotel/lodging semantic similarity use case."""

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path

# Test cases for hotel/lodging matching
TEST_CASES = [
    # (query, candidate, expected_high_similarity)
    ("lodging: hotel check-in (Lodging)", "Hotel Gracery Shinjuku | hotel, lodging | Shinjuku, Japan", True),
    ("lodging: hotel check-in (Lodging)", "hotel, lodging", True),
    ("lodging: hotel check-in (Lodging)", "Hotel Gracery | hotel, lodging", True),
    ("lodging: hotel check-in (Lodging)", "Meiji Shrine", False),
    ("lodging: hotel check-in (Lodging)", "Ueno Park", False),
    ("lodging: hotel check-in (Lodging)", "Shinjuku Japan tourism", False),
    ("lodging: hotel check-in (Lodging)", "accommodation booking", True),

    # Additional semantic tests
    ("food: dinner at restaurant (Dining)", "Sushi Dai | restaurant, food | Tsukiji, Japan", True),
    ("food: dinner at restaurant (Dining)", "Tokyo Tower observation deck", False),
    ("attraction: visit museum (Sightseeing)", "Tokyo National Museum | museum, culture | Ueno, Japan", True),
    ("attraction: visit museum (Sightseeing)", "hotel check-in", False),
]

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_model(model_path: Path):
    """Test a single embedding model."""
    model_name = model_path.stem.replace("_Embedding", "")
    tokenizer_path = model_path.parent / f"{model_name}_tokenizer"

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load model
    try:
        model = ct.models.MLModel(str(model_path))
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        return None

    results = []

    for query, candidate, expected_high in TEST_CASES:
        # Tokenize
        q_tokens = tokenizer(query, return_tensors="np", padding=True, truncation=True, max_length=512)
        c_tokens = tokenizer(candidate, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Get embeddings
        try:
            q_emb = model.predict({
                "inputIds": q_tokens["input_ids"].astype(np.int32),
                "attentionMask": q_tokens["attention_mask"].astype(np.float32)
            })["embeddings"]

            c_emb = model.predict({
                "inputIds": c_tokens["input_ids"].astype(np.int32),
                "attentionMask": c_tokens["attention_mask"].astype(np.float32)
            })["embeddings"]
        except Exception as e:
            print(f"  ERROR running inference: {e}")
            return None

        sim = cosine_similarity(q_emb, c_emb)
        results.append((query, candidate, sim, expected_high))

    # Print results
    print(f"\n  {'Query':<40} | {'Candidate':<50} | Sim   | Exp")
    print(f"  {'-'*40}-+-{'-'*50}-+-------+-----")

    correct = 0
    total = len(results)

    for q, c, sim, expected in results:
        # Consider >0.5 as "similar" for evaluation
        is_high = sim > 0.5
        match = "✓" if is_high == expected else "✗"
        if is_high == expected:
            correct += 1
        exp_str = "High" if expected else "Low"
        print(f"  {q[:40]:<40} | {c[:50]:<50} | {sim:.3f} | {exp_str} {match}")

    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")

    # Key metric: similarity difference between hotel text with/without location
    # r = (query, candidate, sim, expected)
    hotel_with_loc = next((r[2] for r in results if "Shinjuku, Japan" in str(r[1])), 0)
    hotel_without_loc = next((r[2] for r in results if str(r[1]).strip() == "hotel, lodging"), 0)
    loc_penalty = hotel_without_loc - hotel_with_loc

    print(f"\n  Key metrics:")
    print(f"    'hotel, lodging' similarity: {hotel_without_loc:.3f}")
    print(f"    With location (Shinjuku): {hotel_with_loc:.3f}")
    print(f"    Location penalty: {loc_penalty:.3f}")

    return {
        "name": model_name,
        "accuracy": accuracy,
        "hotel_bare": hotel_without_loc,
        "hotel_with_loc": hotel_with_loc,
        "loc_penalty": loc_penalty,
    }

def main():
    models_dir = Path("models")
    embedding_models = sorted(models_dir.glob("*_Embedding.mlpackage"))

    print(f"Found {len(embedding_models)} embedding models")

    all_results = []
    for model_path in embedding_models:
        result = test_model(model_path)
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Best models for location-robust semantic matching")
    print("="*80)
    print(f"\n{'Model':<35} | Accuracy | Hotel+Loc | Hotel Only | Loc Penalty")
    print(f"{'-'*35}-+----------+-----------+------------+------------")

    # Sort by lowest location penalty (least affected by location terms)
    all_results.sort(key=lambda x: -x["hotel_with_loc"])  # Highest sim with location = best

    for r in all_results:
        print(f"{r['name']:<35} | {r['accuracy']:>6.0f}%  | {r['hotel_with_loc']:>9.3f} | {r['hotel_bare']:>10.3f} | {r['loc_penalty']:>10.3f}")

    print("\n✓ Best model = highest 'Hotel+Loc' score (understands category despite location)")

if __name__ == "__main__":
    main()
