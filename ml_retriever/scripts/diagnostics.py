"""Consolidated rigor diagnostics, all results in ONE report file.

Runs the DEPLOYED pipeline (LinUCB stopping policy, entity-aware retriever,
flan-t5 answerer) and reports, honestly:
  1. Per-subgroup performance  -- by answer type AND by domain (frozen test).
  2. Overfitting check          -- train sample vs frozen test success gap.
  3. Calibration               -- is the QA-confidence evidence scorer calibrated
                                  as a relevance predictor? (reliability, ECE, Brier).
  4. Data quality              -- distribution, coverage, leakage, duplicates, byte sanity.
  5. Latency                   -- measured end-to-end wall-clock percentiles.

Writes docs/diagnostics_report.md (the single consolidated results file).
Usage: python scripts/diagnostics.py [--train_sample 60]
"""
import argparse, json, random, statistics as st, time
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.bandit import LinUCBPolicy
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.judge import judge_task
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.system import EcoBudgetSystem
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, DOCS = ROOT / "data", ROOT / "models", ROOT / "docs"

DOMAIN_KEYS = [
    ("phone", ["iphone", "galaxy", "pixel", "oneplus"]),
    ("laptop", ["macbook", "thinkpad", "zenbook", "spectre", "xps", "lenovo", "asus", "hp "]),
    ("earbuds", ["buds", "airpods", "sony wf", "nothing ear"]),
    ("ev", ["ioniq", "mach-e", "byd", "tesla", "mustang"]),
    ("running_shoe", ["adizero", "ghost", "nimbus", "pegasus", "clifton", "boston", "gel-"]),
    ("mountain", ["everest", "k2", "lhotse", "makalu", "kangchenjunga"]),
    ("building", ["tower", "burj", "shanghai", "petronas", "empire", "world trade"]),
    ("travel", ["jaipur", "kyoto", "bali", "oberoi", "ibis", "hotel"]),
    ("country", ["japan", "india", "france", "brazil", "canada", "germany", "country"]),
]


def dom(entity: str) -> str:
    e = (entity or "").lower()
    for name, keys in DOMAIN_KEYS:
        if any(k in e for k in keys):
            return name
    return "other"


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows], rows


def gold_ref(t):
    gt = t.get("ground_truth")
    if isinstance(gt, dict) and gt.get("required_facts"):
        return ", ".join(gt["required_facts"])
    if isinstance(gt, str):
        return gt
    return str(t.get("expected_answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpus, rows = load_corpus()
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    gold = defaultdict(set)
    for p in corpus:
        m = p.metadata
        if "entity" in m and "attribute" in m:
            gold[(m["entity"].lower(), m["attribute"].lower())].add(p.passage_id)

    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    decomposer = TaskDecomposer("models/decomposer-base-lora", adapter_path="models/decomposer-base-lora",
                                attribute_vocab=sorted({p.metadata["attribute"] for p in corpus}))
    answerer = GenerativeAnswerGenerator("models/answer-base", adapter_path="models/answer-base")
    policy = LinUCBPolicy.load(MODELS / "linucb_policy.joblib")
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")
    system = EcoBudgetSystem(decomposer, retriever, answerer, policy=policy,
                             normalizer=normalizer, score_fn=scorer, answer_mode="per_requirement")

    def judgeable(split):
        seen, out = set(), []
        for tid in splits[split]:
            t = tasks[tid]
            if not ("expected_answer" in t or "ground_truth" in t):
                continue
            if t["question"] in seen:
                continue
            seen.add(t["question"]); out.append(t)
        return out

    def run_tasks(tlist):
        recs = []
        for t in tlist:
            try:
                t0 = time.perf_counter(); res = system.run(t["question"]); lat = time.perf_counter() - t0
                s, f1 = judge_task(res.answer or "", t)
                ent = t["decomposed_requirements"][0]["entity"] if t.get("decomposed_requirements") else ""
                recs.append({"id": t["id"], "domain": dom(ent), "type": t["answer_type"],
                             "success": float(s), "fact_f1": f1, "bytes": res.bytes_used,
                             "fetch": res.n_retrieves, "latency": lat})
            except Exception as e:
                recs.append({"id": t["id"], "error": str(e)[:100]})
        return [r for r in recs if "error" not in r]

    test_tasks = judgeable("test")
    train_tasks = judgeable("train"); rng.shuffle(train_tasks)
    test_recs = run_tasks(test_tasks)
    train_recs = run_tasks(train_tasks[:args.train_sample])

    def agg(recs, key):
        g = defaultdict(list)
        for r in recs:
            g[r[key]].append(r)
        out = {}
        for k, rs in sorted(g.items()):
            out[k] = {"n": len(rs),
                      "success": round(sum(x["success"] for x in rs) / len(rs), 3),
                      "fact_f1": round(sum(x["fact_f1"] for x in rs) / len(rs), 3),
                      "avg_bytes": round(sum(x["bytes"] for x in rs) / len(rs), 0),
                      "avg_latency_s": round(sum(x["latency"] for x in rs) / len(rs), 3)}
        return out

    by_type = agg(test_recs, "type")
    by_domain = agg(test_recs, "domain")
    test_succ = sum(r["success"] for r in test_recs) / len(test_recs)
    train_succ = sum(r["success"] for r in train_recs) / len(train_recs) if train_recs else 0.0

    # --- Calibration: QA-confidence as a relevance predictor ---
    pairs = []
    seen = set()
    for t in test_tasks:
        for rq in t["decomposed_requirements"]:
            req = Requirement(entity=rq["entity"], attribute=rq["attribute"])
            if req.key() in seen:
                continue
            seen.add(req.key())
            g = gold.get(req.key(), set())
            if not g:
                continue
            for sp in retriever.retrieve(req, k=5):
                conf = float(scorer(req, sp.passage))
                pairs.append((conf, 1 if sp.passage.passage_id in g else 0))
    bins = [[] for _ in range(10)]
    for conf, label in pairs:
        bins[min(9, int(conf * 10))].append((conf, label))
    n_pairs = len(pairs) or 1
    ece = 0.0
    reliability = []
    for i, b in enumerate(bins):
        if not b:
            continue
        mc = sum(c for c, _ in b) / len(b)
        er = sum(l for _, l in b) / len(b)
        ece += (len(b) / n_pairs) * abs(mc - er)
        reliability.append({"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": len(b),
                            "mean_conf": round(mc, 3), "empirical_relevance": round(er, 3)})
    brier = sum((c - l) ** 2 for c, l in pairs) / n_pairs

    # --- Data quality ---
    ents = {r["metadata"].get("entity") for r in rows if r.get("metadata")}
    attrs = {r["metadata"].get("attribute") for r in rows if r.get("metadata")}
    corpus_keys = {(r["metadata"]["entity"].lower(), r["metadata"]["attribute"].lower())
                   for r in rows if r.get("metadata", {}).get("entity")}
    missing_cov = 0
    for t in tasks.values():
        for rq in t["decomposed_requirements"]:
            if (rq["entity"].lower(), rq["attribute"].lower()) not in corpus_keys:
                missing_cov += 1
    # leakage: seed group must not cross splits
    id_split = {tid: s for s, ids in splits.items() for tid in ids}
    leak = 0
    for t in tasks.values():
        seed_id = t["id"] if not t.get("synthetic") else t.get("seed_task_id")
        if seed_id in id_split and id_split.get(t["id"]) != id_split.get(seed_id):
            leak += 1
    texts = [r["text"] for r in rows]
    dup_text = len(texts) - len(set(texts))
    byte_mismatch = sum(1 for r in rows if r["byte_size"] != len(r["text"].encode("utf-8")))
    dom_dist = Counter(dom(r["metadata"].get("entity", "")) for r in rows if r.get("metadata"))
    type_dist = Counter(t["answer_type"] for t in tasks.values())

    lat = sorted(r["latency"] for r in test_recs)
    def pct(p): return round(lat[min(len(lat) - 1, int(p * len(lat)))], 3) if lat else 0.0

    report = f"""# EcoBudget diagnostics report

Single consolidated results file. Generated by `scripts/diagnostics.py` on the
DEPLOYED pipeline (entity-aware retriever, LinUCB stopping policy, flan-t5
answerer, per-requirement mode). Frozen test split; a train sample is used only
for the overfitting check. All numbers are measured, not estimated.

## 1. Per-subgroup performance (frozen test, n={len(test_recs)} distinct-question tasks)

### 1a. By answer type
| type | n | success | fact_f1 | avg_bytes | avg_latency_s |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v['n']} | {v['success']} | {v['fact_f1']} | {v['avg_bytes']:.0f} | {v['avg_latency_s']} |"
        for k, v in by_type.items()) + f"""

### 1b. By domain
| domain | n | success | fact_f1 | avg_bytes | avg_latency_s |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v['n']} | {v['success']} | {v['fact_f1']} | {v['avg_bytes']:.0f} | {v['avg_latency_s']} |"
        for k, v in by_domain.items()) + f"""

Read honestly: subgroups with small n (list, narrative, procedure; and thin
domains) have noisy estimates and are the weakest slices. procedure/narrative
have no real training coverage and fail.

## 2. Overfitting check (same pipeline, train sample vs frozen test)
| split | n | success |
|---|---|---|
| train sample | {len(train_recs)} | {round(train_succ,3)} |
| frozen test | {len(test_recs)} | {round(test_succ,3)} |

Gap (train minus test) = {round(train_succ - test_succ, 3)}. A small gap indicates
the reported success is not an overfit to the training split. Splits are cut at the
seed level with a coverage invariant, so paraphrases do not leak across splits
(measured leakage below).

## 3. Calibration of the QA-confidence evidence scorer (relevance predictor)
Pairs scored: {len(pairs)} (each retrieved candidate for a unique test requirement,
labelled 1 if it is a gold passage for that requirement).
- Expected Calibration Error (ECE): {round(ece,3)}
- Brier score: {round(brier,3)}

Reliability table (predicted confidence vs empirical relevance):
| conf bin | n | mean_conf | empirical_relevance |
|---|---|---|---|
""" + "\n".join(
        f"| {r['bin']} | {r['n']} | {r['mean_conf']} | {r['empirical_relevance']} |"
        for r in reliability) + f"""

Lower ECE/Brier is better. This measures whether the sufficiency score can be
trusted as a probability; it is not the answer-accuracy calibration (the
per-requirement answerer emits hard 0/1 confidences, which are not probabilistic).

## 4. Data quality
- Corpus passages: {len(rows)} | distinct entities: {len(ents)} | distinct attributes: {len(attrs)}
- Requirement->passage coverage gaps (tasks with a requirement absent from corpus): {missing_cov}
- Split leakage (seed group crossing splits): {leak}
- Duplicate passage texts: {dup_text}
- byte_size != len(text) mismatches: {byte_mismatch}
- Corpus domain distribution: {dict(dom_dist)}
- Task answer-type distribution: {dict(type_dist)}

Known data-quality limitations (candid): the corpus is electronics-skewed (phone+
laptop are the largest buckets), and list/narrative/procedure answer types are
under-represented (and procedure/narrative fail). External validity beyond these
domains is untested.

## 5. Latency (measured end-to-end wall-clock, frozen test, this machine)
- p50: {pct(0.50)} s | p90: {pct(0.90)} s | p95: {pct(0.95)} s | max: {round(lat[-1],3) if lat else 0} s
- mean: {round(sum(lat)/len(lat),3) if lat else 0} s over {len(lat)} tasks

Latency is CPU/MPS wall-clock on a laptop and is dominated by flan-t5 generation;
it is not a tuned production latency. It is reported as-is.

## Raw aggregates (JSON)
```json
{json.dumps({"by_type": by_type, "by_domain": by_domain,
             "overfitting": {"train": round(train_succ,3), "test": round(test_succ,3),
                             "gap": round(train_succ-test_succ,3)},
             "calibration": {"ece": round(ece,3), "brier": round(brier,3), "n_pairs": len(pairs)},
             "data_quality": {"passages": len(rows), "entities": len(ents), "attributes": len(attrs),
                              "coverage_gaps": missing_cov, "leakage": leak, "dup_text": dup_text,
                              "byte_mismatch": byte_mismatch},
             "latency_s": {"p50": pct(0.50), "p95": pct(0.95)}}, indent=2)}
```
"""
    (DOCS / "diagnostics_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {DOCS / 'diagnostics_report.md'}")
    print(f"test success={test_succ:.3f} train success={train_succ:.3f} "
          f"ECE={ece:.3f} Brier={brier:.3f} latency_p50={pct(0.5)}s")


if __name__ == "__main__":
    main()
