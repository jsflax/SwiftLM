#!/usr/bin/env python3
"""
Trip Planning Fine-Tuning Data Generator V2

Extends trip_finetune_data.py with skeleton and activity planning stages using
real POI data from MapKit.

Pipeline stages:
1. Intent Parsing: User utterance -> ParsedTripIntent
2. Segmentation: ParsedTripIntent -> SegmentSkeletons (sub-destinations)
3. Skeleton Planning: Segment + candidates -> DaySkeleton[] (day structure with slots)
4. Activity Planning: DaySkeleton + candidates -> Itinerary.Day (filled slots)

Uses:
- GPT-4o-mini for generating teacher outputs
- MapKit (via PyObjC) for fetching real candidate places

Usage:
    python trip_finetune_data_v2.py --stage skeleton --count 100 --output skeleton_train.jsonl
    python trip_finetune_data_v2.py --stage activity --count 100 --output activity_train.jsonl
    python trip_finetune_data_v2.py --stage all --count 100 --output all_train.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import time

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

# Import MapKit POI search
try:
    from mapkit_poi import search_activities, search_pois, geocode
    MAPKIT_AVAILABLE = True
except ImportError:
    print("Warning: MapKit POI search not available. Will use synthetic data.")
    MAPKIT_AVAILABLE = False

# ============================================================================
# Constants from existing trip_finetune_data.py
# ============================================================================

TRAVEL_PERSONAS = ["bohemian", "bougie", "classic", "outdoorsy", "nightOwl", "familyFriendly"]
GUIDE_TAGS = ["Foodie", "Art & Culture", "Nightlife", "Outdoors", "Family"]
LODGING_STYLES = ["hotel", "hostel", "boutique", "apartment", "ryokan", "guesthouse"]
BUDGET_TIERS = ["budget", "midrange", "luxury"]

ACTIVITY_CLASSES = [
    "landmark", "museum", "gallery", "park", "hike", "beach", "water", "hotSpring",
    "winter", "view", "cafe", "restaurant", "bar", "nightlife", "shopping", "kid",
    "event", "transit", "lodging", "other"
]

SLOT_LABELS = ["morning", "lunch", "afternoon", "dinner", "evening", "lodging"]

# Destinations that work well for specific interests
INTEREST_DESTINATIONS = {
    "surfing": [
        ("Santa Teresa", "Costa Rica"),
        ("Nosara", "Costa Rica"),
        ("San Juan del Sur", "Nicaragua"),
        ("Bali", "Indonesia"),
        ("Uluwatu", "Indonesia"),
        ("Byron Bay", "Australia"),
        ("Sayulita", "Mexico"),
        ("Tamarindo", "Costa Rica"),
        ("Ericeira", "Portugal"),
        ("Hossegor", "France"),
    ],
    "culture": [
        ("Kyoto", "Japan"),
        ("Rome", "Italy"),
        ("Paris", "France"),
        ("Barcelona", "Spain"),
        ("Istanbul", "Turkey"),
        ("Cairo", "Egypt"),
        ("Athens", "Greece"),
        ("Prague", "Czech Republic"),
    ],
    "food": [
        ("Tokyo", "Japan"),
        ("Bangkok", "Thailand"),
        ("Mexico City", "Mexico"),
        ("Lisbon", "Portugal"),
        ("Singapore", "Singapore"),
        ("San Sebastian", "Spain"),
        ("Bologna", "Italy"),
        ("Oaxaca", "Mexico"),
    ],
    "hiking": [
        ("Queenstown", "New Zealand"),
        ("Cusco", "Peru"),
        ("Chamonix", "France"),
        ("Torres del Paine", "Chile"),
        ("Zermatt", "Switzerland"),
        ("Banff", "Canada"),
        ("Hakone", "Japan"),
    ],
    "beach": [
        ("Maldives", "Maldives"),
        ("Zanzibar", "Tanzania"),
        ("Phuket", "Thailand"),
        ("Tulum", "Mexico"),
        ("Mykonos", "Greece"),
        ("Amalfi Coast", "Italy"),
        ("Seychelles", "Seychelles"),
    ],
}

# Default interests by persona
PERSONA_INTERESTS = {
    "bohemian": ["street art", "local markets", "live music", "coffee shops", "vintage shopping"],
    "bougie": ["fine dining", "wine tasting", "luxury spas", "art galleries", "rooftop bars"],
    "classic": ["landmarks", "museums", "historical sites", "local cuisine", "city walks"],
    "outdoorsy": ["hiking", "surfing", "wildlife", "national parks", "adventure sports"],
    "nightOwl": ["nightlife", "bars", "live music", "late-night eats", "clubs"],
    "familyFriendly": ["parks", "beaches", "kid activities", "family restaurants", "zoos"],
}


# ============================================================================
# Skeleton Planning Prompts
# ============================================================================

SKELETON_SYSTEM_PROMPT = """You create daily SKELETON plans for a travel segment.

Given:
- segment: {name, days, segmentStartIdx}
- selectedGuides: tag preferences (Foodie, Outdoors, etc.)
- persona: travel style (bohemian, bougie, classic, etc.)
- interests: specific activities they want (surfing, ramen, etc.)
- candidates: POIs with name, categories
- supplySummary: counts of each activity type available

Create a DaySkeleton for each day with:
- dayIndex: segment.segmentStartIdx + offset
- summary: Brief description of the day's theme
- slots: Array of {label, intent, tags}
  - label: "morning"|"lunch"|"afternoon"|"dinner"|"evening"|"lodging"
  - intent: Short, concrete activity intent (e.g., "surf session", "ramen lunch")
  - tags: Relevant persona/guide tags
- focusInterests: Which interests this day emphasizes (optional)
- feasibilityHint: Notes on what's available (optional)

CRITICAL Rules:
1. PRIORITIZE user's interests - if they want surfing, multiple surf sessions per day
2. For specialized trips, the primary activity should dominate (70-80% of slots)
3. Use "lodging" slot for check-in (first day) and check-out (last day)
4. 3-5 slots per day (nightOwl gets 5, others get 4)
5. Intents must be concrete: "morning surf at Playa Hermosa" not "water activity"

Output JSON Schema:
{
    "days": [{
        "dayIndex": int,
        "summary": string,
        "slots": [{"label": string, "intent": string, "tags": [string]}],
        "focusInterests": [string] | null,
        "feasibilityHint": string | null
    }]
}"""

# ============================================================================
# Activity Planning Prompts
# ============================================================================

ACTIVITY_SYSTEM_PROMPT = """You fill skeleton slots with actual places from candidates.

Given:
- segment: destination name
- skeleton: DaySkeleton with slots and intents
- candidatePlaces: Array of {id, name, categories, latitude, longitude, address}
- persona, lodgingStyle, lodgingBudget

For each slot in skeleton.slots:
1. Find the best matching place from candidatePlaces
2. Create an activity with:
   - placeId: ID of the selected place
   - placeName: Name of the selected place
   - startTime: Approximate time (e.g., "09:00", "12:30")
   - duration: Estimated duration in minutes
   - notes: Brief activity description
3. If slot.alternateIntents exist and primary can't be fulfilled, try alternates

CRITICAL:
- Only use places from candidatePlaces (use their actual IDs)
- Match intents to appropriate place categories
- Consider travel time between consecutive activities
- Lodging slots should use hotel/lodging candidates

Output JSON Schema:
{
    "dayIndex": int,
    "date": string | null,
    "activities": [{
        "placeId": string,
        "placeName": string,
        "slotLabel": string,
        "startTime": string,
        "duration": int,
        "notes": string
    }]
}"""


# ============================================================================
# Place/POI Generation
# ============================================================================

def fetch_pois_for_destination(destination: str, country: str, interests: List[str], limit: int = 30) -> List[Dict]:
    """Fetch real POIs from MapKit for a destination."""
    if not MAPKIT_AVAILABLE:
        return generate_synthetic_pois(destination, interests, limit)

    location = f"{destination}, {country}"
    all_pois = []
    seen = set()

    # Search for each interest
    for interest in interests[:5]:  # Limit to avoid too many API calls
        try:
            results = search_activities(interest, location, radius_km=25)
            for r in results:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    all_pois.append({
                        "id": f"poi_{len(all_pois)}",
                        "name": r["name"],
                        "categories": [r.get("category", interest)],
                        "latitude": r["latitude"],
                        "longitude": r["longitude"],
                        "address": r.get("address", ""),
                    })
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  Warning: Failed to search for {interest} in {location}: {e}")

    # Also search for general categories
    for category in ["restaurant", "cafe", "hotel", "park", "museum"]:
        if len(all_pois) >= limit:
            break
        try:
            results = search_pois(category, location_name=location, limit=10)
            for r in results:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    all_pois.append({
                        "id": f"poi_{len(all_pois)}",
                        "name": r["name"],
                        "categories": [r.get("category", category)],
                        "latitude": r["latitude"],
                        "longitude": r["longitude"],
                        "address": r.get("address", ""),
                    })
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: Failed to search for {category}: {e}")

    return all_pois[:limit]


def generate_synthetic_pois(destination: str, interests: List[str], limit: int = 30) -> List[Dict]:
    """Generate synthetic POI data when MapKit is unavailable."""
    pois = []

    # Generate interest-related POIs
    for interest in interests:
        for i in range(3):
            pois.append({
                "id": f"poi_{len(pois)}",
                "name": f"{destination} {interest.title()} Spot {i+1}",
                "categories": [interest],
                "latitude": random.uniform(-90, 90),
                "longitude": random.uniform(-180, 180),
                "address": f"Main Street {random.randint(1, 100)}, {destination}",
            })

    # Add standard POIs
    for cat, names in [
        ("restaurant", ["Cafe Central", "Local Kitchen", "Street Food Market"]),
        ("hotel", ["Grand Hotel", "Boutique Stay", "Surf Lodge"]),
        ("park", ["Central Park", "Botanical Gardens"]),
        ("museum", ["National Museum", "Art Gallery"]),
    ]:
        for name in names:
            pois.append({
                "id": f"poi_{len(pois)}",
                "name": f"{destination} {name}",
                "categories": [cat],
                "latitude": random.uniform(-90, 90),
                "longitude": random.uniform(-180, 180),
                "address": f"Avenue {random.randint(1, 50)}, {destination}",
            })

    return pois[:limit]


def build_supply_summary(candidates: List[Dict]) -> Dict:
    """Build a supply summary from candidates (mirrors Swift's buildSupplySummary)."""
    totals = {}
    exemplars = {}

    cat_to_class = {
        "restaurant": "restaurant", "cafe": "cafe", "bar": "bar", "hotel": "lodging",
        "museum": "museum", "gallery": "gallery", "park": "park", "beach": "beach",
        "surfing": "water", "hiking": "hike", "landmark": "landmark", "shopping": "shopping",
        "nightlife": "nightlife", "spa": "other", "zoo": "kid", "aquarium": "kid",
        "store": "shopping", "mkpoicategorystore": "shopping", "mkpoicategoryrestaurant": "restaurant",
        "mkpoicategorycafe": "cafe", "mkpoicategoryhotel": "lodging", "mkpoicategorymuseum": "museum",
        "mkpoicategorypark": "park", "mkpoicategorybeach": "beach",
    }

    for p in candidates:
        categories = p.get("categories", ["other"])
        for cat in categories:
            if cat is None:
                cat = "other"
            cls = cat_to_class.get(cat.lower(), "other")
            totals[cls] = totals.get(cls, 0) + 1
            if cls not in exemplars:
                exemplars[cls] = []
            if len(exemplars[cls]) < 3:
                exemplars[cls].append(p.get("name", "Unknown"))

    return {"totals": totals, "topExemplars": exemplars}


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_skeleton_example(
    client: OpenAI,
    destination: str,
    country: str,
    days: int,
    interests: List[str],
    persona: str,
    guides: List[str],
    candidates: List[Dict],
    model: str = "gpt-4o-mini"
) -> Optional[Dict]:
    """Generate a skeleton planning training example."""
    try:
        segment = {
            "name": destination,
            "regionHint": None,
            "days": days,
            "segmentStartIdx": 0,
        }

        supply_summary = build_supply_summary(candidates)

        input_data = {
            "segment": segment,
            "selectedGuides": guides,
            "persona": persona,
            "lodgingStyle": random.choice(LODGING_STYLES),
            "lodgingBudget": random.choice(BUDGET_TIERS),
            "interests": interests,
            "allowedLabels": SLOT_LABELS,
            "targetSlotsPerDay": 5 if persona == "nightOwl" else 4,
            "candidates": candidates[:20],  # Truncate for context
            "supplySummary": supply_summary,
            "shouldCheckIn": True,
            "shouldCheckOut": True,
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SKELETON_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)}
            ],
            response_format={"type": "json_object"}
        )

        output = json.loads(response.choices[0].message.content)

        return {
            "messages": [
                {"role": "system", "content": SKELETON_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)},
                {"role": "assistant", "content": json.dumps(output)}
            ],
            "metadata": {
                "stage": "skeleton",
                "destination": destination,
                "country": country,
                "days": days,
                "interests": interests,
            }
        }
    except Exception as e:
        print(f"Error generating skeleton example: {e}")
        return None


def generate_activity_example(
    client: OpenAI,
    destination: str,
    skeleton_day: Dict,
    candidates: List[Dict],
    persona: str,
    model: str = "gpt-4o-mini"
) -> Optional[Dict]:
    """Generate an activity planning training example."""
    try:
        input_data = {
            "segment": destination,
            "skeleton": skeleton_day,
            "candidatePlaces": candidates[:20],
            "persona": persona,
            "lodgingStyle": random.choice(LODGING_STYLES),
            "lodgingBudget": random.choice(BUDGET_TIERS),
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACTIVITY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)}
            ],
            response_format={"type": "json_object"}
        )

        output = json.loads(response.choices[0].message.content)

        return {
            "messages": [
                {"role": "system", "content": ACTIVITY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_data)},
                {"role": "assistant", "content": json.dumps(output)}
            ],
            "metadata": {
                "stage": "activity",
                "destination": destination,
                "dayIndex": skeleton_day.get("dayIndex", 0),
            }
        }
    except Exception as e:
        print(f"Error generating activity example: {e}")
        return None


def generate_full_pipeline_example(client: OpenAI) -> Optional[Dict]:
    """Generate a complete pipeline example: intent -> segment -> skeleton -> activity."""
    # Pick a random interest and its destinations
    interest_type = random.choice(list(INTEREST_DESTINATIONS.keys()))
    destination, country = random.choice(INTEREST_DESTINATIONS[interest_type])

    # Generate parameters
    persona = random.choice(TRAVEL_PERSONAS)
    interests = random.sample(PERSONA_INTERESTS.get(persona, ["food", "culture"]), 2)
    interests.insert(0, interest_type)  # Primary interest first

    guides = random.sample(GUIDE_TAGS, random.randint(1, 3))
    days = random.choice([3, 4, 5, 7])

    print(f"  Generating for {destination}, {country} ({interest_type}, {days} days)...")

    # Fetch POIs
    candidates = fetch_pois_for_destination(destination, country, interests)
    if len(candidates) < 5:
        print(f"    Warning: Only {len(candidates)} POIs found, using synthetic data")
        candidates = generate_synthetic_pois(destination, interests)

    print(f"    Found {len(candidates)} POIs")

    # Generate skeleton
    skeleton_example = generate_skeleton_example(
        client, destination, country, days, interests, persona, guides, candidates
    )
    if not skeleton_example:
        return None

    # Extract skeleton days for activity planning
    skeleton_output = json.loads(skeleton_example["messages"][-1]["content"])
    skeleton_days = skeleton_output.get("days", [])

    if not skeleton_days:
        print("    Warning: No skeleton days generated")
        return None

    # Generate activity for first day (to keep examples manageable)
    activity_example = generate_activity_example(
        client, destination, skeleton_days[0], candidates, persona
    )

    return {
        "skeleton": skeleton_example,
        "activity": activity_example,
        "destination": destination,
        "country": country,
        "interests": interests,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate skeleton/activity planning training data")
    parser.add_argument("--stage", choices=["skeleton", "activity", "all"],
                       default="all", help="Which stage to generate data for")
    parser.add_argument("--count", type=int, default=50, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="trip_train_v2.jsonl", help="Output file path")
    parser.add_argument("--dry-run", action="store_true", help="Test POI fetching without GPT calls")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run: Testing POI fetching...\n")
        for interest, destinations in list(INTEREST_DESTINATIONS.items())[:2]:
            dest, country = destinations[0]
            print(f"\n{interest.upper()}: {dest}, {country}")
            pois = fetch_pois_for_destination(dest, country, [interest, "food", "hotel"])
            print(f"  Found {len(pois)} POIs:")
            for p in pois[:5]:
                cats = [c for c in p.get('categories', ['unknown']) if c]
                print(f"    - {p['name']} ({', '.join(cats) if cats else 'unknown'})")
        return

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)
    output_path = Path(args.output)

    print(f"Generating {args.count} {args.stage} examples...")
    print(f"MapKit available: {MAPKIT_AVAILABLE}")
    print(f"Output: {output_path}\n")

    examples = []
    failed = 0

    for i in range(args.count):
        print(f"[{i+1}/{args.count}] ", end="", flush=True)

        example = generate_full_pipeline_example(client)
        if example:
            examples.append(example)
            print(f"  OK ({len(examples)} generated)")
        else:
            failed += 1
            print(f"  FAILED")

        # Rate limiting
        if (i + 1) % 5 == 0:
            time.sleep(2)

    # Write output
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone! Generated {len(examples)} examples ({failed} failed)")
    print(f"Output written to: {output_path}")

    # Also write separate files
    if args.stage == "all":
        skeleton_path = output_path.with_suffix(".skeleton.jsonl")
        activity_path = output_path.with_suffix(".activity.jsonl")

        with open(skeleton_path, "w") as f:
            for ex in examples:
                if ex.get("skeleton"):
                    f.write(json.dumps(ex["skeleton"]) + "\n")

        with open(activity_path, "w") as f:
            for ex in examples:
                if ex.get("activity"):
                    f.write(json.dumps(ex["activity"]) + "\n")

        print(f"Also wrote: {skeleton_path}")
        print(f"Also wrote: {activity_path}")


if __name__ == "__main__":
    main()
