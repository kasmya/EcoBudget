"""Idea 1: pick the energy-aware reward weight mu on the VAL split (never test).

For each mu, train an energy-reward LinUCB on train, evaluate on val, and report
success / fetches / 5G radio energy (fast-dormancy) so we can choose the mu that
preserves task success at the lowest energy. Test is left frozen.

Usage: python scripts/sweep_mu_energy.py --mus 0.05 0.1 0.15 0.2 0.3
"""
import argparse
import json
import random
from pathlib import Path

import joblib

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import LinUCBPolicy
from ml_retriever.energy import FiveGTransferModel, PayloadModel, RadioStateModel
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import build_candidates, run_episode
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"
HL = {"comparison", "single_fact", "multi_part", "yes_no"}
_PL, _TX = PayloadModel(), FiveGTransferModel()
_RFD = RadioStateModel(demote_between_fetches=True)


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows]


def judgeable(split):
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    return [tasks[i] for i in splits[split] if "expected_answer" in tasks[i] or "ground_truth" in tasks[i]]


def reqs_of(t):
    return [Requirement(entity=r["entity"], attribute=r["attribute"]) for r in t["decomposed_requirements"]]


def radio_fd_j(n_fetch):
    if n_fetch <= 0:
        return 0.0
    b = n_fetch * _PL.html_page_bytes
    return _RFD.account(b, n_fetch)["radio_j"] + _TX.energy_joules(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mus", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = load_corpus()
    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")

    train = judgeable("train")
    val = judgeable("val")
    tr_c = {t["id"]: build_candidates(reqs_of(t), retriever, k=5) for t in train}
    va_c = {t["id"]: build_candidates(reqs_of(t), retriever, k=5) for t in val}

    print(f"{'mu':>6}{'succ':>8}{'succ_HL':>9}{'fetch':>7}{'bytes':>8}{'radioJ_fd':>11}")
    for mu in args.mus:
        pol = LinUCBPolicy(alpha=1.0, seed=args.seed)
        rng = random.Random(args.seed)
        for ep in range(args.epochs):
            order = list(train); rng.shuffle(order)
            for t in order:
                res = run_episode(t, tr_c[t["id"]], scorer, ag,
                                  lambda c, s: pol.select_action(normalizer.transform(c), explore=True),
                                  reward_mode="energy", mu=mu, threshold=0.3)
                for c, a, r in res.trajectory:
                    pol.update(normalizer.transform(c), a, r)
        # eval on val (greedy)
        s = sHL = nHL = f = b = e = 0.0
        for t in val:
            res = run_episode(t, va_c[t["id"]], scorer, ag,
                              lambda c, st: pol.select_action(normalizer.transform(c), explore=False),
                              reward_mode="energy", mu=mu, threshold=0.3)
            s += res.success; f += res.n_retrieves; b += res.total_bytes
            e += radio_fd_j(res.n_retrieves)
            if t["answer_type"] in HL:
                sHL += res.success; nHL += 1
        n = len(val)
        print(f"{mu:>6.2f}{s/n:>8.3f}{sHL/max(nHL,1):>9.3f}{f/n:>7.2f}{b/n:>8.0f}{e/n:>11.2f}")


if __name__ == "__main__":
    main()
