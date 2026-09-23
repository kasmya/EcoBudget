"""Re-measure the frozen test set with VERDICT-AWARE scoring for yes_no and
comparison (not mere required-fact presence). Writes docs/verdict_eval.md.

For each yes_no task: compute the ground-truth yes/no from the two corpus values
and the question direction, and the verdict the system implies from the values it
actually emitted; success = they match. For 'which is X' comparison tasks with
numeric values: same, on the winning entity. Descriptive cases (no direction word
or non-numeric values) fall back to the existing fact-coverage judge, and are
reported separately so nothing is hidden.

Usage: python scripts/verdict_eval.py
"""
import json
from collections import defaultdict
from pathlib import Path

import joblib

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.bandit import LinUCBPolicy
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.judge import judge_task
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.system import EcoBudgetSystem
from ml_retriever.types import Passage
from ml_retriever.verdict import (comparison_ground_truth, yes_no_ground_truth,
                                   yes_no_predicted, all_magnitudes, parse_magnitude,
                                   question_direction)

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, DOCS = ROOT / "data", ROOT / "models", ROOT / "docs"


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows], rows


def main():
    corpus, rows = load_corpus()
    cval = {}
    for r in rows:
        m = r.get("metadata", {})
        if "entity" in m and "attribute" in m:
            cval[(m["entity"].lower(), m["attribute"].lower())] = r["text"]

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())

    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    decomposer = TaskDecomposer("models/decomposer-base-lora", adapter_path="models/decomposer-base-lora",
                                attribute_vocab=sorted({p.metadata["attribute"] for p in corpus}))
    answerer = GenerativeAnswerGenerator("models/answer-base", adapter_path="models/answer-base")
    policy = LinUCBPolicy.load(MODELS / "linucb_policy.joblib")
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")
    system = EcoBudgetSystem(decomposer, retriever, answerer, policy=policy,
                             normalizer=normalizer, score_fn=scorer, answer_mode="per_requirement")

    seen, test_tasks = set(), []
    for tid in splits["test"]:
        t = tasks[tid]
        if ("expected_answer" in t or "ground_truth" in t) and t["question"] not in seen:
            seen.add(t["question"]); test_tasks.append(t)

    # counters + exact per-task honest score (verdict where defined, else fact-coverage)
    cnt = defaultdict(lambda: {"n": 0, "fact": 0, "verdict": 0, "verdict_n": 0, "honest": 0})
    examples = []
    for t in test_tasks:
        res = system.run(t["question"])
        ans = res.answer or ""
        typ = t["answer_type"]
        fact_ok, _ = judge_task(ans, t)
        c = cnt[typ]; c["n"] += 1; c["fact"] += int(fact_ok)

        verdict_ok = None
        reqs = t.get("decomposed_requirements", [])
        if typ == "yes_no" and len(reqs) == 2:
            e1, e2 = reqs[0]["entity"], reqs[1]["entity"]
            v1 = cval.get((e1.lower(), reqs[0]["attribute"].lower()), "")
            v2 = cval.get((e2.lower(), reqs[1]["attribute"].lower()), "")
            truth = yes_no_ground_truth(t["question"], v1, v2, e1, e2)
            pred = yes_no_predicted(t["question"], ans, e1, e2)
            if truth is not None and pred is not None:
                verdict_ok = (pred == truth)
        elif typ == "comparison" and len(reqs) == 2:
            e1, e2 = reqs[0]["entity"], reqs[1]["entity"]
            v1 = cval.get((e1.lower(), reqs[0]["attribute"].lower()), "")
            v2 = cval.get((e2.lower(), reqs[1]["attribute"].lower()), "")
            truth = comparison_ground_truth(t["question"], e1, v1, e2, v2)
            mags = all_magnitudes(ans, [e1, e2]); d = question_direction(t["question"])
            if truth is not None and len(mags) >= 2 and d is not None:
                win_first = (mags[0] > mags[1]) if d == "greater" else (mags[0] < mags[1])
                verdict_ok = ((e1 if win_first else e2) == truth)

        if verdict_ok is not None:
            c["verdict_n"] += 1; c["verdict"] += int(verdict_ok)
            c["honest"] += int(verdict_ok)  # verdict-scored task
            if len(examples) < 12:
                examples.append((t["question"][:60], ans[:24], "OK" if verdict_ok else "WRONG"))
        else:
            c["honest"] += int(fact_ok)      # fall back to fact-coverage

    def line(typ):
        c = cnt[typ]
        fa = c["fact"] / c["n"] if c["n"] else 0
        if c["verdict_n"]:
            return f"| {typ} | {c['n']} | {fa:.3f} | {c['verdict']}/{c['verdict_n']} = {c['verdict']/c['verdict_n']:.3f} |"
        return f"| {typ} | {c['n']} | {fa:.3f} | n/a (fact-coverage) |"

    honest_num = sum(c["honest"] for c in cnt.values())
    honest_den = sum(c["n"] for c in cnt.values())

    rep = "# Verdict-aware re-measurement (frozen test)\n\n"
    rep += ("Yes/no and 'which is X' comparison tasks are scored on the CORRECT VERDICT "
            "(computed from the two corpus values and the question direction, checked against "
            "the verdict implied by the values the system emitted), not on required-fact "
            "presence. Descriptive/non-numeric cases fall back to fact-coverage and are marked.\n\n")
    rep += "| type | n | fact-coverage (old) | verdict accuracy (new) |\n|---|---|---|---|\n"
    for typ in ("yes_no", "comparison", "single_fact", "multi_part", "list", "narrative", "procedure"):
        if cnt[typ]["n"]:
            rep += line(typ) + "\n"
    yn = cnt["yes_no"]; cm = cnt["comparison"]
    rep += (f"\n**Headline shift.** yes_no fact-coverage "
            f"{yn['fact']/max(yn['n'],1):.3f} -> verdict accuracy "
            f"{yn['verdict']/max(yn['verdict_n'],1):.3f} on {yn['verdict_n']} numeric yes/no tasks. "
            f"comparison verdict accuracy {cm['verdict']}/{cm['verdict_n']}.\n")
    rep += (f"\n**Honest overall success (verdict where defined, else fact-coverage): "
            f"{honest_num/honest_den:.3f}** over {honest_den} tasks. "
            f"(The old fact-coverage-only headline was 0.895.)\n\n")
    rep += "Sample verdict checks (question | emitted values | verdict):\n\n"
    for q, a, ok in examples:
        rep += f"- {ok}: {q!r} -> {a!r}\n"
    (DOCS / "verdict_eval.md").write_text(rep, encoding="utf-8")
    print(f"Wrote {DOCS/'verdict_eval.md'}")
    print(f"yes_no verdict acc={yn['verdict']}/{yn['verdict_n']} "
          f"comparison verdict acc={cm['verdict']}/{cm['verdict_n']} "
          f"honest_overall={honest_num/honest_den:.3f}")


if __name__ == "__main__":
    main()
