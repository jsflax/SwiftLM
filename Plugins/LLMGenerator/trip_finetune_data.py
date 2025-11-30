#!/usr/bin/env python3
"""
Trip Planning Fine-Tuning Data Generator

Generates training data for fine-tuning Qwen 1.5B on JoyJet's trip planning pipeline.
Uses GPT-4 to generate high-quality input/output pairs that can be distilled to smaller models.

Stages covered:
1. Intent Parsing: User utterance -> ParsedTripIntent
2. Segmentation: ParsedTripIntent -> SegmentSkeletons

Usage:
    python trip_finetune_data.py --stage intent --count 100 --output intent_train.jsonl
    python trip_finetune_data.py --stage segmentation --count 100 --output segment_train.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional
import os

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

# ============================================================================
# Schema Definitions (matching JoyJet's Swift types)
# ============================================================================

TRAVEL_PERSONAS = ["bohemian", "bougie", "classic", "outdoorsy", "nightOwl", "familyFriendly"]
GUIDE_TAGS = ["Foodie", "Art & Culture", "Nightlife", "Outdoors", "Family"]
LODGING_STYLES = ["hotel", "hostel", "boutique", "apartment", "ryokan", "guesthouse"]
BUDGET_TIERS = ["$", "$$", "$$$"]

# Sample destinations for variety - expanded for better coverage
DESTINATIONS = {
    "countries": [
        # Asia
        "Japan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Philippines",
        "Malaysia", "Singapore", "Taiwan", "Cambodia", "Sri Lanka", "Nepal",
        # Europe
        "Italy", "France", "Spain", "Portugal", "Greece", "Croatia", "Turkey",
        "Iceland", "Norway", "Sweden", "Denmark", "Netherlands", "Germany",
        "Switzerland", "Austria", "Czech Republic", "Ireland", "Scotland", "England",
        # Americas
        "Costa Rica", "Peru", "Mexico", "Brazil", "Argentina", "Colombia", "Ecuador",
        "Chile", "Nicaragua", "Panama", "Guatemala", "Belize", "Cuba",
        # Oceania & Africa
        "Australia", "New Zealand", "Fiji", "Morocco", "South Africa", "Egypt",
        "Kenya", "Tanzania"
    ],
    "cities": [
        # Asia
        "Tokyo", "Kyoto", "Osaka", "Bangkok", "Chiang Mai", "Ho Chi Minh City",
        "Hanoi", "Singapore", "Hong Kong", "Seoul", "Taipei", "Bali",
        # Europe
        "Paris", "Rome", "Barcelona", "Lisbon", "Amsterdam", "Prague", "Vienna",
        "Berlin", "London", "Edinburgh", "Dublin", "Florence", "Venice", "Seville",
        "Madrid", "Porto", "Athens", "Dubrovnik", "Budapest", "Copenhagen", "Stockholm",
        "Reykjavik", "Munich", "Zurich",
        # Americas
        "New York", "San Francisco", "Los Angeles", "Miami", "Austin", "New Orleans",
        "Seattle", "Portland", "Denver", "Chicago", "Boston", "San Diego",
        "Honolulu", "Mexico City", "Tulum", "Cancun", "Lima", "Buenos Aires",
        "Medellin", "Cartagena", "Rio de Janeiro",
        # Oceania
        "Sydney", "Melbourne", "Auckland", "Queenstown"
    ],
    "regions": [
        # Beach/Surf destinations
        "Hawaii", "Maui", "Oahu", "Big Island", "Kauai", "Bali", "Maldives",
        "Costa Rica Pacific Coast", "Nicoya Peninsula", "Santa Teresa",
        "Puerto Escondido", "Sayulita", "Cabo San Lucas", "Punta Mita",
        "Caribbean", "Bahamas", "Turks and Caicos", "Barbados", "Aruba",
        "Riviera Maya", "Yucatan Peninsula",
        # Nature/Adventure
        "Tuscany", "Provence", "Okinawa", "Patagonia", "Amalfi Coast",
        "Scottish Highlands", "Swiss Alps", "Greek Islands", "Santorini", "Mykonos",
        "Algarve", "Cinque Terre", "Lake Como", "Dolomites", "Norwegian Fjords",
        "Galapagos Islands", "Amazon", "Torres del Paine",
        # US regions
        "California Coast", "Pacific Northwest", "Florida Keys", "Outer Banks",
        "Napa Valley", "Sonoma", "Big Sur", "Lake Tahoe", "Yellowstone",
        "Grand Canyon", "Yosemite", "Zion", "Joshua Tree"
    ]
}

INTERESTS = [
    # Food
    "food", "street food", "fine dining", "cooking classes", "wine tasting", "ramen", "sushi",
    "tapas", "local cuisine", "food markets", "cafe hopping",
    # Outdoors
    "surfing", "hiking", "diving", "snorkeling", "beaches", "mountains", "wildlife",
    "national parks", "camping", "kayaking", "cycling", "skiing", "rock climbing",
    # Culture
    "museums", "art galleries", "architecture", "history", "temples", "castles",
    "local culture", "photography", "music", "festivals", "theater",
    # Urban
    "nightlife", "bars", "clubs", "live music", "shopping", "boutiques", "vintage shops",
    "street art", "urban exploration", "rooftop bars",
    # Other
    "wellness", "spa", "yoga", "meditation", "onsen", "hot springs"
]

TRAVEL_STYLES = ["solo", "couple", "friends", "family", "honeymoon", "backpacking", "luxury"]

# ============================================================================
# Utterance Templates for Generating Diverse Inputs
# ============================================================================

UTTERANCE_TEMPLATES = [
    # Simple
    "I want to go to {destination} for {duration}",
    "{duration} in {destination}",
    "Planning a trip to {destination}",
    "Help me plan {duration} in {destination}",

    # With interests
    "I want to visit {destination} for {duration}, interested in {interests}",
    "{duration} trip to {destination} - love {interests}",
    "Going to {destination} for {duration}. I'm really into {interests}",
    "{destination} for {duration}, focusing on {interests}",

    # With style
    "{travel_style} trip to {destination} for {duration}",
    "Planning a {travel_style} {duration} in {destination}",
    "{duration} {travel_style} adventure in {destination}",

    # Complex
    "{travel_style} trip to {destination} for {duration}, interested in {interests}",
    "I want to spend {duration} in {destination} as a {travel_style} traveler, love {interests}",
    "Planning {duration} in {destination}. {travel_style} trip, interested in {interests}. {constraints}",

    # Natural language
    "My partner and I want to go to {destination} for our honeymoon, about {duration}. We love {interests}",
    "Looking for a {duration} getaway to {destination}. Big fan of {interests}",
    "Thinking about {destination} for {duration}. Want to experience {interests}",
    "Need help planning {duration} in {destination} - we're {travel_style} travelers who enjoy {interests}",
]

CONSTRAINTS = [
    "budget-conscious",
    "prefer walking",
    "no early mornings",
    "avoiding tourist traps",
    "wheelchair accessible",
    "with a toddler",
    "vegetarian friendly",
    "prefer public transit",
    "luxury only",
    "off the beaten path"
]


def generate_random_utterance() -> dict:
    """Generate a random travel utterance with metadata."""
    template = random.choice(UTTERANCE_TEMPLATES)

    # Pick destination type
    dest_type = random.choice(["countries", "cities", "regions"])
    destination = random.choice(DESTINATIONS[dest_type])

    # Duration
    duration_days = random.choice([3, 4, 5, 7, 10, 14, 21, 30])
    duration_strs = [
        f"{duration_days} days",
        f"{duration_days//7} weeks" if duration_days >= 7 and duration_days % 7 == 0 else f"{duration_days} days",
        "a week" if duration_days == 7 else f"{duration_days} days",
        "two weeks" if duration_days == 14 else f"{duration_days} days",
    ]
    duration = random.choice(duration_strs)

    # Interests (1-4)
    num_interests = random.randint(1, 4)
    interests_list = random.sample(INTERESTS, num_interests)
    interests = ", ".join(interests_list[:-1]) + " and " + interests_list[-1] if len(interests_list) > 1 else interests_list[0]

    # Travel style
    travel_style = random.choice(TRAVEL_STYLES)

    # Constraints
    constraints = random.choice(CONSTRAINTS) if random.random() > 0.7 else ""

    utterance = template.format(
        destination=destination,
        duration=duration,
        interests=interests,
        travel_style=travel_style,
        constraints=constraints
    ).strip()

    return {
        "utterance": utterance,
        "destination": destination,
        "destination_type": {"countries": "country", "cities": "city", "regions": "region"}[dest_type],
        "duration_days": duration_days,
        "interests": interests_list,
        "travel_style": travel_style,
        "constraints": constraints if constraints else None
    }


# ============================================================================
# GPT-4 Data Generation
# ============================================================================

INTENT_SYSTEM_PROMPT = """You extract travel intent from a single user message.
- Normalize destinations as names ("Japan", "Tokyo", "Osaka", "Kyoto", etc.)
- Set durationDays when user implies time ("2 weeks" -> 14).
- Set destinationDays for each destination for what you deem appropriate for the full trip
- Map interests to high-level words (e.g., "good food", "sushi" -> "food"; "urban exploration" -> "urban exploration"; "surfing" -> "surfing"). Assume a multitude of interests based on intent, travelPersona, and guideTags. Should be a minimum of 3 interests.
- Also suggest guideTags from: ["Foodie","Art & Culture","Nightlife","Outdoors","Family"].
- Return strict JSON only.

Output JSON Schema:
{
    "destinations": [{"name": string, "country": string, "kind": "country"|"city"|"region", "priority": int, "destinationDays": int, "destinationStartDayIndex": int}],
    "durationDays": int,
    "interests": [string],
    "guideTags": [string],
    "constraints": [string] | null,
    "travelStyle": string | null,
    "notes": string | null,
    "travelPersona": "bohemian"|"bougie"|"classic"|"outdoorsy"|"nightOwl"|"familyFriendly" | null,
    "name": string
}"""

SEGMENTATION_SYSTEM_PROMPT = """You are a travel routing planner that divides a destination into logical sub-destinations.

Rules:
- Always start from the root destination the user mentioned.
- For countries or large regions, choose 3–7 major areas or cities that form a coherent route (include arrival and departure cities).
- For single cities, choose 3–6 representative neighborhoods or wards.
- Each subdestination gets an estimated number of days (total ≈ trip length).
- Include a one-line description and travel order (south→north, coastal→inland, etc.).

CRITICAL - Interest Prioritization:
- If the user has specific interests (e.g., "surfing", "hiking", "wildlife"), PRIORITIZE locations that support those activities
- Allocate MORE days to interest-focused locations (e.g., surf towns for surf trips, national parks for wildlife trips)
- For specialized trips (surf, dive, ski, etc.), choose 70-80% of destinations specifically for that activity
- Variety is good (20-30% can be cultural/other), but primary interests must dominate
- Example: "surf trip in Nicaragua" → focus on Pacific surf towns (San Juan del Sur, Popoyo, Playa Maderas), maybe 1-2 days in Granada for variety

Output JSON Schema:
{
    "skeletons": [{
        "name": string,
        "country": string,
        "kind": "city"|"region"|"neighborhood"|"island",
        "days": int,
        "order": int,
        "description": string,
        "candidateActivitySearchRangeInKm": float
    }]
}"""


def generate_intent_example(client: OpenAI, utterance_data: dict, model: str = "gpt-4o-mini") -> Optional[dict]:
    """Generate a single intent parsing training example using GPT."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "utterance": utterance_data["utterance"],
                    "localeHint": "en-US"
                })}
            ],
            response_format={"type": "json_object"}
        )

        output = json.loads(response.choices[0].message.content)

        # Build training example in chat format
        return {
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "utterance": utterance_data["utterance"],
                    "localeHint": "en-US"
                })},
                {"role": "assistant", "content": json.dumps(output)}
            ],
            "metadata": {
                "stage": "intent",
                "source_utterance": utterance_data["utterance"],
                "expected_destination": utterance_data["destination"],
                "expected_duration": utterance_data["duration_days"]
            }
        }
    except Exception as e:
        print(f"Error generating intent example: {e}")
        return None


def generate_segmentation_example(client: OpenAI, intent: dict, model: str = "gpt-4o-mini") -> Optional[dict]:
    """Generate a single segmentation training example using GPT."""
    try:
        # Build segmentation input from intent
        input_data = {
            "destination": intent.get("destinations", [{}])[0].get("name", "Unknown"),
            "tripLengthDays": intent.get("durationDays", 7),
            "persona": intent.get("travelPersona"),
            "interests": intent.get("interests", [])
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)}
            ],
            response_format={"type": "json_object"}
        )

        output = json.loads(response.choices[0].message.content)

        return {
            "messages": [
                {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)},
                {"role": "assistant", "content": json.dumps(output)}
            ],
            "metadata": {
                "stage": "segmentation",
                "destination": input_data["destination"],
                "duration": input_data["tripLengthDays"],
                "interests": input_data["interests"]
            }
        }
    except Exception as e:
        print(f"Error generating segmentation example: {e}")
        return None


def generate_combined_example(client: OpenAI) -> Optional[dict]:
    """Generate a combined intent+segmentation example (end-to-end)."""
    utterance_data = generate_random_utterance()

    # First get intent
    intent_example = generate_intent_example(client, utterance_data)
    if not intent_example:
        return None

    # Parse the intent output
    intent_output = json.loads(intent_example["messages"][-1]["content"])

    # Then get segmentation
    seg_example = generate_segmentation_example(client, intent_output)
    if not seg_example:
        return None

    return {
        "intent": intent_example,
        "segmentation": seg_example,
        "source_utterance": utterance_data["utterance"]
    }


# ============================================================================
# Main Generation Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate trip planning fine-tuning data")
    parser.add_argument("--stage", choices=["intent", "segmentation", "combined"],
                       default="combined", help="Which stage to generate data for")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="trip_train.jsonl", help="Output file path")
    parser.add_argument("--dry-run", action="store_true", help="Generate utterances without calling GPT-4")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.dry_run:
        # Just generate and print sample utterances
        print("Sample utterances (dry run):\n")
        for i in range(min(10, args.count)):
            data = generate_random_utterance()
            print(f"{i+1}. {data['utterance']}")
            print(f"   -> {data['destination']} ({data['destination_type']}), {data['duration_days']} days")
            print(f"   -> Interests: {data['interests']}")
            print()
        return

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Export your API key: export OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)

    print(f"Generating {args.count} {args.stage} examples...", flush=True)
    print(f"Output: {output_path}", flush=True)

    examples = []
    failed = 0

    for i in range(args.count):
        if args.stage == "intent":
            utterance_data = generate_random_utterance()
            example = generate_intent_example(client, utterance_data)
        elif args.stage == "segmentation":
            # For standalone segmentation, we need to first generate an intent
            utterance_data = generate_random_utterance()
            intent_example = generate_intent_example(client, utterance_data)
            if intent_example:
                intent_output = json.loads(intent_example["messages"][-1]["content"])
                example = generate_segmentation_example(client, intent_output)
            else:
                example = None
        else:  # combined
            example = generate_combined_example(client)

        if example:
            examples.append(example)
            print(f"  [{len(examples)}/{args.count}] Generated example", flush=True)
        else:
            failed += 1
            print(f"  [FAIL] Failed to generate example", flush=True)

        # Rate limiting - be nice to the API
        if (i + 1) % 10 == 0:
            import time
            time.sleep(1)

    # Write output
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone! Generated {len(examples)} examples ({failed} failed)")
    print(f"Output written to: {output_path}")

    # Also write separate files for each stage if combined
    if args.stage == "combined":
        intent_path = output_path.with_suffix(".intent.jsonl")
        seg_path = output_path.with_suffix(".segmentation.jsonl")

        with open(intent_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex["intent"]) + "\n")

        with open(seg_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex["segmentation"]) + "\n")

        print(f"Also wrote: {intent_path}")
        print(f"Also wrote: {seg_path}")


if __name__ == "__main__":
    main()
