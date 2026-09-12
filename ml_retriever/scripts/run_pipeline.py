"""Phase 6 validation: run the full EcoBudgetSystem end-to-end on val tasks.

Decompose -> retrieve-under-(bandit)-policy -> generate answer -> measure bytes,
with NO ground truth in the loop. Reports, per task: the decomposed
requirements, the synthesized answer, bytes, and abstention; plus aggregate
auxiliary quality (token-F1 of the generated answer vs the display gold) and a
bytes comparison between the bandit policy and the heuristic. Every run records
component versions (decomposer / answer_generator / policy).

Usage: python scripts/run_pipeline.py --n 10
"""
import argparse
import json
from pathlib import Path

import joblib

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.bandit import BanditPolicy
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.judge import judge_task, token_f1
from ml_retriever.retriever import RequirementRetriever
from ml_retriever.system import EcoBudgetSystem
from ml_retriever.types import Passage

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    embedding=r.get("embedding"), metadata=r.get("metadata", {})) for r in rows]


def gold_answer(t):
    if t.get("expected_answer"):
        return str(t["expected_answer"])
    gt = t.get("ground_truth")
    if isinstance(gt, str):
        return gt
    if isinstance(gt, dict):
        return ", ".join(gt.get("required_facts", []))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    corpus = load_corpus()
    retriever = RequirementRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    decomposer = TaskDecomposer("models/decomposer-base-lora", adapter_path="models/decomposer-base-lora",
                                attribute_vocab=sorted({p.metadata["attribute"] for p in corpus}))
    answerer = GenerativeAnswerGenerator("models/answer-small", adapter_path="models/answer-small")
    policy = BanditPolicy.load(MODELS / "bandit_policy.joblib")
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    val = [tasks[i] for i in splits["val"] if ("expected_answer" in tasks[i] or "ground_truth" in tasks[i])]
    # de-dup to distinct questions for a readable 10-task run; seeds first
    seen, picked = set(), []
    for t in sorted(val, key=lambda t: t.get("synthetic", False)):
        if t["question"] in seen:
            continue
        seen.add(t["question"]); picked.append(t)
        if len(picked) >= args.n:
            break

    bandit_sys = EcoBudgetSystem(decomposer, retriever, answerer, policy=policy,
                                 normalizer=normalizer, score_fn=scorer)
    heur_sys = EcoBudgetSystem(decomposer, retriever, answerer, score_fn=scorer)

    f1_sum = succ = bytes_bandit = bytes_heur = 0.0
    print(f"{'='*70}\nEnd-to-end run ({len(picked)} tasks)\n{'='*70}")
    for t in picked:
        res = bandit_sys.run(t["question"])
        hres = heur_sys.run(t["question"])
        gold = gold_answer(t)
        ans = res.answer or ""
        f1 = token_f1(ans, gold)
        s, _ = judge_task(ans, t)
        f1_sum += f1; succ += s; bytes_bandit += res.bytes_used; bytes_heur += hres.bytes_used
        print(f"\n[{t['id']}/{t['answer_type']}] {t['question']}")
        print(f"  reqs: {[(r.entity, r.attribute) for r in res.requirements]}")
        print(f"  answer: {ans!r}  (abstained={res.abstained})")
        print(f"  gold:   {gold!r}")
        print(f"  judge_success={bool(s)} token_f1={f1:.2f} | bytes bandit={res.bytes_used} heuristic={hres.bytes_used}")

    n = len(picked)
    print(f"\n{'='*70}")
    print(f"avg token_f1={f1_sum/n:.3f} | judge success={succ/n:.3f} | "
          f"avg bytes: bandit={bytes_bandit/n:.0f} heuristic={bytes_heur/n:.0f}")
    print(f"versions: {bandit_sys.run.__self__ and ''}{res.versions}")


if __name__ == "__main__":
    main()
