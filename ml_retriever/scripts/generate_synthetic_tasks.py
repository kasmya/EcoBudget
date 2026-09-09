"""Expands data/tasks_seed.json into data/tasks.json via deterministic,
template-based paraphrasing -- NOT an LLM call. This is the "generate
synthetic variations of manually-written seed tasks" approach from the
plan's review addendum, used to reach the 150-200 task target without a
proportional increase in manual annotation.

Every synthetic task keeps the exact same `decomposed_requirements`,
`expected_answer` / `ground_truth`, `topic`, `difficulty`, and
`answer_type` as its seed -- only the question wording changes, so ground
truth can never silently drift from the source task. Each synthetic task
also carries `synthetic: true` and `seed_task_id` so later phases (and the
train/val/test split) can group or filter on it.

Run: python scripts/generate_synthetic_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTRIBUTE_PHRASES = {
    "price": "price",
    "battery": "battery capacity",
    "display_refresh_rate": "display refresh rate",
    "weight": "weight",
    "chip": "chip",
    "camera": "main camera resolution",
    "display": "display",
    "best_time_to_visit": "recommended travel season",  # avoids "...to visit of Jaipur"
    "daily_budget": "daily travel budget",
    "top_attraction": "top attraction",
    "typical_trip_length": "typical trip length",
    "price_per_night": "price per night",
    "rating": "guest rating",
    "amenities": "amenities",
    "room_count": "room count",
    "battery_life": "battery life",
    "noise_cancellation": "noise cancellation",
    "codec_support": "audio codec support",
    "height": "height",
    "floors": "floor count",
    "completion_year": "completion year",
    "architect": "architect",
    "elevation": "elevation",
    "location": "location",
    "first_ascent_year": "year of first ascent",
    "population": "population",
    "area": "land area",
    "capital": "capital",
    "official_language": "official language",
    "cushioning": "cushioning",
    "best_for": "recommended use",
    "range": "driving range",
    "battery_capacity": "battery capacity",
}

SINGLE_FACT_TEMPLATES = [
    "What is the {attr} of {e}?",
    "Can you tell me the {attr} of {e}?",
    "How would you describe the {attr} of {e}?",
    "I'd like to know the {attr} of {e}.",
    "Please tell me {e}'s {attr}.",
    "What's {e}'s {attr}?",
    "Do you know the {attr} of {e}?",
    "Give me the {attr} for {e}.",
    "What {attr} does {e} have?",
    "Looking for the {attr} of {e} -- what is it?",
    "Could you look up the {attr} of {e}?",
    "I need the {attr} of {e} for a comparison.",
    "What does {e} offer in terms of {attr}?",
]

COMPARISON_1ATTR_TEMPLATES = [
    "Compare {e1} and {e2} on {attr}.",
    "Which is better in terms of {attr}: {e1} or {e2}?",
    "How does {e1}'s {attr} compare to {e2}'s?",
    "Between {e1} and {e2}, which has the better {attr}?",
    "{e1} vs {e2}: which wins on {attr}?",
    "I'm deciding between {e1} and {e2} -- which has the better {attr}?",
    "Can you compare {attr} between {e1} and {e2}?",
    "In terms of {attr}, how do {e1} and {e2} differ?",
    "Of {e1} and {e2}, which has better {attr}?",
]

COMPARISON_2ATTR_TEMPLATES = [
    "Compare {e1} and {e2} on {attr1} and {attr2}.",
    "How do {e1} and {e2} differ in {attr1} and {attr2}?",
    "I'm choosing between {e1} and {e2} -- how do they compare on {attr1} and {attr2}?",
    "Which is the better choice, {e1} or {e2}, considering {attr1} and {attr2}?",
    "Break down {e1} vs {e2} by {attr1} and {attr2}.",
    "What are the differences between {e1} and {e2} in {attr1} and {attr2}?",
    "Help me weigh {e1} against {e2} on {attr1} and {attr2}.",
]


def _phrase(attribute: str) -> str:
    return ATTRIBUTE_PHRASES.get(attribute, attribute.replace("_", " "))


def _requirement_shape(reqs: list[dict]):
    entities = list(dict.fromkeys(r["entity"] for r in reqs))
    attributes = list(dict.fromkeys(r["attribute"] for r in reqs))
    return entities, attributes


def generate_variants(task: dict) -> list[str]:
    """Generates up to N_VARIANTS_PER_FAMILY paraphrases per seed task.

    With 15 seeds this used the full template lists (8-10 variants each)
    to help reach the 150-200 target. After adding 21 more seeds to cover
    travel, product selection, specification lookup, multi-attribute
    research, decision making, and sustainability (see plan.md /
    codebase feedback on corpus breadth), using the full template lists
    for all 36 seeds would overshoot 200. Capping at 4 variants per seed
    keeps the total in range while still giving each seed multiple
    paraphrases.
    """
    N_VARIANTS_PER_FAMILY = 4

    reqs = task["decomposed_requirements"]
    entities, attributes = _requirement_shape(reqs)

    if len(reqs) == 1:
        e = entities[0]
        attr = _phrase(attributes[0])
        templates = SINGLE_FACT_TEMPLATES[:N_VARIANTS_PER_FAMILY]
        return [t.format(e=e, attr=attr) for t in templates]

    if len(entities) == 2 and len(attributes) == 1:
        e1, e2 = entities
        attr = _phrase(attributes[0])
        templates = COMPARISON_1ATTR_TEMPLATES[:N_VARIANTS_PER_FAMILY]
        return [t.format(e1=e1, e2=e2, attr=attr) for t in templates]

    if len(entities) == 2 and len(attributes) == 2:
        e1, e2 = entities
        attr1, attr2 = (_phrase(a) for a in attributes)
        templates = COMPARISON_2ATTR_TEMPLATES[:N_VARIANTS_PER_FAMILY]
        return [
            t.format(e1=e1, e2=e2, attr1=attr1, attr2=attr2)
            for t in templates
        ]

    # Fallback for shapes not covered by a template family: no synthetic
    # variants rather than a wrong-shaped guess.
    return []


def build(seed_tasks: list[dict]) -> list[dict]:
    all_tasks: list[dict] = []
    for task in seed_tasks:
        base = dict(task)
        base["synthetic"] = False
        all_tasks.append(base)

        variants = generate_variants(task)
        for i, question in enumerate(variants, start=1):
            variant = dict(task)
            variant["id"] = f"{task['id']}-s{i:02d}"
            variant["question"] = question
            variant["synthetic"] = True
            variant["seed_task_id"] = task["id"]
            all_tasks.append(variant)
    return all_tasks


def main() -> None:
    seed_path = ROOT / "data" / "tasks_seed.json"
    seed_tasks = json.loads(seed_path.read_text(encoding="utf-8"))

    all_tasks = build(seed_tasks)

    out_path = ROOT / "data" / "tasks.json"
    out_path.write_text(json.dumps(all_tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    n_synthetic = sum(1 for t in all_tasks if t["synthetic"])
    print(f"Wrote {len(all_tasks)} tasks to {out_path} "
          f"({len(seed_tasks)} seed + {n_synthetic} synthetic)")


if __name__ == "__main__":
    main()
