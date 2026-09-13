"""
Turns a Context into a fixed-length numeric feature vector for the
bandit. Deliberately excludes anything eval-only (ground truth).

Feature order is fixed and documented here because LinUCB's weight
vector has no other way of telling you what each dimension means.

Note on estimated_remaining_information: this field in Context now
carries marginal confidence gain (how much did the last retrieval step
improve confidence?), not a step countdown. It is computable from real
retrieval responses without access to task internals. When it is near
zero or negative, further retrieval is unlikely to help — that is the
signal the bandit should learn to act on.
"""
from typing import List

from .context import Context

FEATURE_NAMES = [
    "data_used_kb",
    "resources_seen",
    "relevance_score",
    "answer_confidence",
    "evidence_coverage",
    "steps_taken",
    "marginal_confidence_gain",   # renamed from estimated_remaining_information
    "relevance_delta",
    "confidence_delta",
]


class ContextVectorBuilder:
    """
    Stateful across a single episode: tracks previous relevance/confidence
    to compute deltas. Call reset() at the start of each new episode.
    """

    def __init__(self):
        self._prev_relevance = None
        self._prev_confidence = None

    def reset(self):
        self._prev_relevance = None
        self._prev_confidence = None

    def build(self, context: Context) -> List[float]:
        relevance_delta = (
            0.0 if self._prev_relevance is None
            else context.relevance_score - self._prev_relevance
        )
        confidence_delta = (
            0.0 if self._prev_confidence is None
            else context.answer_confidence - self._prev_confidence
        )

        self._prev_relevance = context.relevance_score
        self._prev_confidence = context.answer_confidence

        return [
            context.data_used / 1000.0,
            float(context.resources_seen),
            context.relevance_score,
            context.answer_confidence,
            context.evidence_coverage,
            float(context.steps_taken),
            context.estimated_remaining_information or 0.0,  # marginal confidence gain
            relevance_delta,
            confidence_delta,
        ]