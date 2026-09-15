"""Phase 6: build (prompt -> answer) pairs to fine-tune the generative answerer.

Input = the ANSWER_PROMPT filled with the question and the task's GOLD evidence
(the corpus passages matching each requirement's (entity, attribute)); target =
the display answer the model should synthesize (expected_answer when present --
e.g. a comparison verdict -- else the string/joined-required_facts gold).

Train/val only (test stays frozen). narrative/list included so the generator
learns those forms too. Writes data/answer_{train,val}.jsonl.
"""
import json
from pathlib import Path

from ml_retriever.answer import ANSWER_PROMPT
from ml_retriever.types import Requirement

DATA = Path(__file__).resolve().parent.parent / "data"


def target_answer(task) -> str:
    if task.get("expected_answer"):
        return str(task["expected_answer"])
    gt = task.get("ground_truth")
    if isinstance(gt, str):
        return gt
    if isinstance(gt, dict) and gt.get("required_facts"):
        return ", ".join(gt["required_facts"])
    return ""


def main():
    corpus = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    by_key = {}
    for r in corpus:
        m = r.get("metadata", {})
        if "entity" in m and "attribute" in m:
            by_key.setdefault((m["entity"].lower(), m["attribute"].lower()), []).append(r["text"])

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())

    for split in ("train", "val"):
        rows = []
        for tid in splits[split]:
            t = tasks[tid]
            target = target_answer(t)
            if not target:
                continue
            evidence_texts = []
            for r in t["decomposed_requirements"]:
                key = (r["entity"].lower(), r["attribute"].lower())
                evidence_texts.extend(by_key.get(key, []))
            if not evidence_texts:
                continue
            prompt = ANSWER_PROMPT.format(question=t["question"],
                                          evidence=" ".join(evidence_texts)[:2000])
            rows.append({"task_id": tid, "input": prompt, "target": target})
        out = DATA / f"answer_{split}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} {split} pairs to {out}")


if __name__ == "__main__":
    main()
