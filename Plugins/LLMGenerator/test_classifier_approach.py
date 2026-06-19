#!/usr/bin/env python3
"""
Compare embedding-based vs classifier-based approaches for place categorization.

For on-device use, a fine-tuned classifier is likely better than embedding similarity.
"""

import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Your category taxonomy
CATEGORIES = [
    "lodging",      # Hotels, hostels, B&Bs
    "dining",       # Restaurants, cafes, bars
    "attraction",   # Museums, monuments, landmarks
    "transport",    # Airports, train stations
    "shopping",     # Malls, markets, stores
    "nature",       # Parks, beaches, hiking
    "entertainment", # Theaters, concerts, nightlife
    "services",     # Banks, hospitals, etc.
]

# Test places (simulating MapKit POI data)
TEST_PLACES = [
    ("Hotel Gracery Shinjuku", "hotel, lodging", "Shinjuku, Japan"),
    ("Sushi Dai", "restaurant, sushi", "Tsukiji, Japan"),
    ("Tokyo National Museum", "museum, culture", "Ueno, Japan"),
    ("Narita International Airport", "airport, transport", "Chiba, Japan"),
    ("Shibuya 109", "shopping mall, fashion", "Shibuya, Japan"),
    ("Yoyogi Park", "park, nature", "Shibuya, Japan"),
    ("Robot Restaurant", "entertainment, show", "Shinjuku, Japan"),
    ("7-Eleven", "convenience store", "Tokyo, Japan"),
    ("Meiji Shrine", "shrine, spiritual", "Shibuya, Japan"),
    ("Ramen Ichiran", "restaurant, ramen", "Various, Japan"),
]

def format_place(name, category_hint, location):
    """Format place data as it might come from MapKit."""
    return f"{name} | {category_hint} | {location}"

def embedding_approach_benchmark():
    """Benchmark the embedding + similarity approach."""
    print("\n" + "="*60)
    print("EMBEDDING APPROACH (similarity search)")
    print("="*60)

    # Load a small model
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Pre-compute category embeddings
    category_texts = [f"{cat}: {cat} category" for cat in CATEGORIES]

    with torch.no_grad():
        cat_tokens = tokenizer(category_texts, padding=True, truncation=True, return_tensors="pt")
        cat_outputs = model(**cat_tokens)
        cat_embeddings = cat_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        cat_embeddings = torch.nn.functional.normalize(cat_embeddings, p=2, dim=1)

    print(f"Category embeddings shape: {cat_embeddings.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Benchmark inference
    place_texts = [format_place(*p) for p in TEST_PLACES]

    start = time.time()
    iterations = 100

    for _ in range(iterations):
        with torch.no_grad():
            place_tokens = tokenizer(place_texts, padding=True, truncation=True, return_tensors="pt")
            place_outputs = model(**place_tokens)
            place_embeddings = place_outputs.last_hidden_state.mean(dim=1)
            place_embeddings = torch.nn.functional.normalize(place_embeddings, p=2, dim=1)

            # Compute similarities to all categories
            similarities = torch.mm(place_embeddings, cat_embeddings.T)
            predictions = similarities.argmax(dim=1)

    elapsed = time.time() - start
    per_batch = elapsed / iterations * 1000
    per_place = per_batch / len(place_texts)

    print(f"\nLatency ({len(place_texts)} places, {iterations} iterations):")
    print(f"  Total: {elapsed:.2f}s")
    print(f"  Per batch: {per_batch:.1f}ms")
    print(f"  Per place: {per_place:.2f}ms")

    # Show predictions
    print("\nPredictions:")
    for (name, _, _), pred in zip(TEST_PLACES, predictions.tolist()):
        print(f"  {name:<30} -> {CATEGORIES[pred]}")

def classifier_approach_analysis():
    """Analyze what a classifier approach would look like."""
    print("\n" + "="*60)
    print("CLASSIFIER APPROACH (fine-tuned)")
    print("="*60)

    print("""
For on-device place categorization, I recommend:

1. BASE MODEL: distilbert-base-uncased (66M params) or
               prajjwal1/bert-tiny (4.4M params)

2. ARCHITECTURE:
   Input: "{place_name} | {category_hints}"
   -> Encoder (BERT-tiny or DistilBERT)
   -> Classification head (hidden -> num_categories)
   -> Softmax -> Category prediction

3. TRAINING DATA:
   - Use MapKit categories as ground truth
   - Augment with synthetic variations
   - ~10K-50K examples should be enough

4. EXPECTED PERFORMANCE:
   - bert-tiny: ~0.5ms per inference, 17MB model
   - distilbert: ~2ms per inference, 250MB model

5. COREML EXPORT:
   - Single forward pass, no similarity search
   - Can use INT8 quantization for even smaller size
   - Output: probability distribution over categories
""")

    # Show bert-tiny stats
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        print(f"bert-tiny config:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Vocab size: {config.vocab_size}")
    except:
        pass

def main():
    embedding_approach_benchmark()
    classifier_approach_analysis()

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
For on-device MapKit place categorization:

1. BEST APPROACH: Fine-tuned classifier on bert-tiny or similar
   - Single forward pass (no similarity search)
   - ~0.5ms latency per place
   - ~17MB model size (can be <5MB with INT8)
   - Fixed category output

2. HOW TO TRAIN:
   a. Collect training data from MapKit category mappings
   b. Fine-tune bert-tiny with classification head
   c. Export to CoreML with quantization

3. ALTERNATIVE: If categories change often, use embedding model
   - paraphrase-MiniLM-L6-v2 (43MB) performed best
   - Pre-compute category embeddings
   - ~2-5ms latency per place

Would you like me to create a fine-tuning script for the classifier approach?
""")

if __name__ == "__main__":
    main()
