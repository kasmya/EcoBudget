"""Phase 4: Evidence tracking with requirements.

Tracks, for a fixed set of Requirements, whether the Passages retrieved so far
provide sufficient evidence for each one -- using ONLY the passages added (no
ground truth, no corpus metadata). Satisfies the Phase 0 `EvidenceTracker`
protocol, so the controller/bandit can treat it as a black box.

Sufficiency is a CONTENT signal, not a metadata match. A passage is evidence
for a requirement to the degree its text is semantically close to the
requirement ("entity attribute"); the default scorer is MiniLM cosine
similarity (reusing a passage's precomputed embedding when present). This is
why a passage that mentions the right entity but never its attribute -- e.g.
"the iPhone 15 weighs 171 g" for the requirement (iPhone 15, price) -- scores
low and does NOT satisfy it, whereas a naive entity-match would wrongly count
it. The scorer is injectable so the test suite can exercise the tracker's
gating logic with a transparent, model-free lexical score.

Design mirrors the rest of the package: the scoring that needs a model is
lazy/injectable; the tracker's state machine is pure and fully unit-tested.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from .retriever import _unit, requirement_to_query
from .types import Passage, Requirement

# A scorer maps (requirement, passage) -> evidence strength, higher = stronger.
ScoreFn = Callable[[Requirement, Passage], float]


class MiniLMScorer:
    """Default evidence scorer: cosine similarity between the requirement's
    query embedding and the passage embedding.

    Reuses `passage.embedding` when present (corpus passages are pre-embedded);
    otherwise embeds the passage text. Requirement query embeddings are cached.
    Lazily imports sentence-transformers so importing this module -- and the
    model-free tests that inject their own scorer -- needs no model download.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._query_cache: dict[tuple[str, str], np.ndarray] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, requirement: Requirement, passage: Passage) -> float:
        key = requirement.key()
        if key not in self._query_cache:
            vec = self._get_model().encode(requirement_to_query(requirement))
            self._query_cache[key] = _unit(np.asarray(vec, dtype=float))
        q = self._query_cache[key]
        emb = passage.embedding
        if emb is None:
            emb = self._get_model().encode(passage.text)
        return float(np.dot(q, _unit(np.asarray(emb, dtype=float))))


class EvidenceCoverageTracker:
    """Per-requirement evidence tracker (Phase 0 `EvidenceTracker` protocol).

    A requirement is SATISFIED once the best evidence score seen for it reaches
    `threshold`. `coverage()` is a continuous partial-credit view (each
    requirement contributes min(best_score/threshold, 1)), distinct from the
    hard-count `frac_satisfied()`.
    """

    def __init__(
        self,
        requirements: Iterable[Requirement],
        threshold: float = 0.5,
        score_fn: Optional[ScoreFn] = None,
    ):
        # dedup by (entity, attribute) key, preserving insertion order
        self.requirements: list[Requirement] = []
        seen_keys: set[tuple[str, str]] = set()
        for r in requirements:
            if r.key() in seen_keys:
                continue
            seen_keys.add(r.key())
            self.requirements.append(r)

        self.threshold = threshold
        self._score_fn: ScoreFn = score_fn or MiniLMScorer()
        self._best_score: dict[tuple[str, str], float] = {r.key(): 0.0 for r in self.requirements}
        self._best_passage_id: dict[tuple[str, str], Optional[str]] = {
            r.key(): None for r in self.requirements
        }
        self._seen_passage_ids: set[str] = set()

    def add_passage(self, passage: Passage) -> None:
        # idempotent per passage_id
        if passage.passage_id in self._seen_passage_ids:
            return
        self._seen_passage_ids.add(passage.passage_id)
        for r in self.requirements:
            score = self._score_fn(r, passage)
            if score > self._best_score[r.key()]:
                self._best_score[r.key()] = score
                self._best_passage_id[r.key()] = passage.passage_id

    def _is_satisfied(self, key: tuple[str, str]) -> bool:
        return self._best_score[key] >= self.threshold

    def is_sufficient(self) -> bool:
        return all(self._is_satisfied(r.key()) for r in self.requirements)

    def frac_satisfied(self) -> float:
        if not self.requirements:
            return 1.0
        return sum(self._is_satisfied(r.key()) for r in self.requirements) / len(self.requirements)

    def frac_remaining(self) -> float:
        return 1.0 - self.frac_satisfied()

    def coverage(self) -> float:
        if not self.requirements:
            return 1.0
        total = sum(
            min(max(self._best_score[r.key()], 0.0) / self.threshold, 1.0)
            for r in self.requirements
        )
        return total / len(self.requirements)

    def unanswered_requirements(self) -> list[Requirement]:
        return [r for r in self.requirements if not self._is_satisfied(r.key())]

    # --- introspection helpers (not part of the protocol) ---
    def best_score(self, requirement: Requirement) -> float:
        return self._best_score.get(requirement.key(), 0.0)

    def evidence_passage_id(self, requirement: Requirement) -> Optional[str]:
        return self._best_passage_id.get(requirement.key())


def gather_evidence(
    requirements: Iterable[Requirement],
    retriever,
    k: int = 5,
    threshold: float = 0.5,
    score_fn: Optional[ScoreFn] = None,
) -> EvidenceCoverageTracker:
    """Wire Phase 3 -> Phase 4: retrieve per requirement and feed the passages
    into a tracker. This is the per-requirement replacement for the old
    whole-question evidence-adding path. Needs a real retriever (and thus a
    model); the tracker's logic itself is covered model-free in the tests.
    """
    requirements = list(requirements)
    tracker = EvidenceCoverageTracker(requirements, threshold=threshold, score_fn=score_fn)
    for req in requirements:
        for scored in retriever.retrieve(req, k=k):
            tracker.add_passage(scored.passage)
    return tracker
