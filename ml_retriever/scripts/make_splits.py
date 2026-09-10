"""Creates data/splits.json: a train/val/test split over data/tasks.json.

Splits at the SEED-TASK level, not the individual-task level: every
synthetic paraphrase of a given seed task is assigned to the same split as
its seed. Splitting paraphrases independently would let near-duplicate
wording of the same underlying question appear in both train and test --
exactly the leakage the plan's review flagged for the corpus/task split in
general. Grouping by seed_task_id closes that specific hole.

Deterministic (fixed seed groups below, no randomness) so this is
reproducible without needing to pin a PRNG. Run once; per plan.md this
split is then FROZEN and must not be touched again until Phase 7.

Run: python scripts/make_splits.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 41 seed task ids -> split, chosen to keep each split's difficulty, topic,
# and answer-type mix roughly representative. Existing assignments are frozen;
# T37-T41 extend the same seed-grouping rule for new answer types.
SEED_SPLIT_ASSIGNMENT = {
    # train (24 seeds, ~2/3)
    "T1": "train", "T2": "train", "T4": "train", "T5": "train", "T6": "train",
    "T8": "train", "T9": "train", "T11": "train", "T13": "train", "T14": "train",
    "T16": "train", "T18": "train", "T19": "train", "T21": "train", "T22": "train",
    "T24": "train", "T25": "train", "T27": "train", "T28": "train", "T30": "train",
    "T31": "train", "T33": "train", "T34": "train", "T36": "train",
    # val (7 seeds)
    "T3": "val", "T7": "val", "T10": "val", "T17": "val", "T20": "val",
    "T26": "val", "T32": "val",
    # test (5 seeds)
    "T12": "test", "T15": "test", "T23": "test", "T29": "test", "T35": "test",
    # General question-type expansion
    "T37": "train", "T38": "val", "T39": "train", "T40": "test", "T41": "val",
}


def main() -> None:
    tasks_path = ROOT / "data" / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        seed_id = task["id"] if not task["synthetic"] else task["seed_task_id"]
        split = SEED_SPLIT_ASSIGNMENT[seed_id]
        splits[split].append(task["id"])

    assert set(SEED_SPLIT_ASSIGNMENT) == {
        t["id"] for t in tasks if not t["synthetic"]
    }, "every seed task must have a split assignment"

    out_path = ROOT / "data" / "splits.json"
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    lock_path = ROOT / "data" / "splits.json.FROZEN"
    lock_path.write_text(
        "This split was re-frozen on 2026-09-09 after the Phase 1 general\n"
        "question-type expansion (41 seed groups; seven answer types). Per\n"
        "plan.md, do not regenerate or edit data/splits.json (or the\n"
        "seed-to-split assignment in make_splits.py) until Phase 7's final run. Phase 2\n"
        "decomposer training uses train only; Phase 3 recall@5 evaluation\n"
        "uses a separate held-out requirement->passage set, not these\n"
        "task splits; Phase 5 bandit training/eval uses train/val only.\n",
        encoding="utf-8",
    )

    for split_name, ids in splits.items():
        print(f"{split_name}: {len(ids)} tasks")
    print(f"Wrote {out_path} and froze it via {lock_path.name}")


if __name__ == "__main__":
    main()
