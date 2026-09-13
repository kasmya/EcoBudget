"""Phase 6: EcoBudgetSystem -- the full pipeline, end to end.

question -> decompose -> requirements -> retrieve-under-policy (STOP/RETRIEVE)
-> evidence -> generate answer -> measure bytes. No ground truth anywhere: this
is the deployed path, distinct from the training/eval rollout that judges
against gold. The answer generator is frozen and its `generator_version` is
recorded on every run for reproducibility (plan.md).

Retrieval reuses the Phase 3 retriever for similarity-aware candidates and the
Phase 5 rollout primitives for the stop/continue loop. If a trained policy is
given it drives retrieval; otherwise the simple sufficiency heuristic does.
Everything is injectable, so the orchestration is unit-tested without models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bandit import RETRIEVE, STOP
from .evidence import EvidenceCoverageTracker, QAScorer
from .rollout import Candidate, EpisodeState, build_context, build_candidates, decide_heuristic
from .types import Passage, Requirement


@dataclass
class SystemResult:
    question: str
    requirements: list[Requirement]
    answer: Optional[str]
    abstained: bool
    confidence: float
    used_evidence_ids: list[str]
    bytes_used: int
    n_retrieves: int
    decisions: list[int]
    versions: dict


class EcoBudgetSystem:
    def __init__(
        self,
        decomposer,
        retriever,
        answer_generator,
        policy=None,
        normalizer=None,
        score_fn=None,
        k: int = 5,
        threshold: float = 0.3,
        max_steps: int = 30,
    ):
        self.decomposer = decomposer
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.policy = policy
        self.normalizer = normalizer
        self.score_fn = score_fn or QAScorer()
        self.k = k
        self.threshold = threshold
        self.max_steps = max_steps

    def _decide(self, ctx, state: EpisodeState) -> int:
        if self.policy is None:
            return decide_heuristic(ctx, state)
        x = self.normalizer.transform(ctx) if self.normalizer is not None else ctx
        return self.policy.select_action(x, explore=False)

    def run(self, question: str) -> SystemResult:
        requirements = self.decomposer.decompose(question)
        candidates = build_candidates(requirements, self.retriever, k=self.k)
        tracker = EvidenceCoverageTracker(requirements, threshold=self.threshold, score_fn=self.score_fn)
        max_bytes = sum(c.passage.byte_size for cs in candidates.values() for c in cs) or 1
        state = EpisodeState(requirements=requirements, candidates=candidates,
                             tracker=tracker, max_bytes=max_bytes)

        decisions: list[int] = []
        while state.step < self.max_steps:
            ctx = build_context(state)
            action = self._decide(ctx, state)
            if action == RETRIEVE and not state.has_retrievable():
                action = STOP
            decisions.append(int(action))
            if action == STOP:
                break
            req = state.retrieval_target()
            cand = state.next_candidate(req)
            tracker.add_passage(cand.passage)
            state.added_ids.add(cand.passage.passage_id)
            state.bytes_used += cand.passage.byte_size
            state.step += 1

        by_id = {c.passage.passage_id: c.passage for cs in candidates.values() for c in cs}
        evidence = [by_id[pid] for pid in state.added_ids if pid in by_id]
        result = self.answer_generator.generate(question, evidence)

        return SystemResult(
            question=question, requirements=requirements,
            answer=result.answer, abstained=result.abstained, confidence=result.confidence,
            used_evidence_ids=result.used_evidence_ids, bytes_used=state.bytes_used,
            n_retrieves=len(state.added_ids), decisions=decisions,
            versions={
                "decomposer": getattr(self.decomposer, "version", "?"),
                "answer_generator": getattr(self.answer_generator, "version", "?"),
                "policy": "bandit" if self.policy is not None else "heuristic",
            },
        )
