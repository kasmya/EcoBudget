"""Model-free tests for Phase 3 retrieval ranking math.

Like the rest of this suite, these use dummy embeddings and never load a
model -- they cover rank_passages / requirement_to_query, the parts that
don't need sentence-transformers. RequirementRetriever's query encoding is
exercised by scripts/eval_retriever.py on a machine with the model.
"""

import math

from ml_retriever.retriever import rank_passages, requirement_to_query
from ml_retriever.types import Passage, Requirement


def _p(pid, emb, byte_size=100, text="x"):
    return Passage(passage_id=pid, text=text, byte_size=byte_size, embedding=emb)


class TestRequirementToQuery:
    def test_deslugs_attribute(self):
        q = requirement_to_query(Requirement(entity="iPhone 15", attribute="display_refresh_rate"))
        assert q == "iPhone 15 display refresh rate"


class TestRankPassages:
    def test_orders_by_cosine_similarity(self):
        # query points along x; p_near is closer in direction than p_far
        passages = [
            _p("far", [0.0, 1.0]),
            _p("near", [1.0, 0.1]),
            _p("mid", [1.0, 1.0]),
        ]
        ranked = rank_passages([1.0, 0.0], passages, k=3)
        assert [r.passage.passage_id for r in ranked] == ["near", "mid", "far"]

    def test_top_k_truncates(self):
        passages = [_p(f"p{i}", [1.0, float(i)]) for i in range(10)]
        ranked = rank_passages([1.0, 0.0], passages, k=3)
        assert len(ranked) == 3

    def test_passages_without_embedding_are_skipped(self):
        passages = [_p("has", [1.0, 0.0]), _p("none", None)]
        ranked = rank_passages([1.0, 0.0], passages, k=5)
        assert [r.passage.passage_id for r in ranked] == ["has"]

    def test_similarity_is_cosine_not_dot(self):
        # same direction, different magnitude -> cosine 1.0 for both
        ranked = rank_passages([1.0, 0.0], [_p("a", [5.0, 0.0])], k=1)
        assert math.isclose(ranked[0].similarity, 1.0, abs_tol=1e-6)

    def test_value_per_byte_prefers_smaller_when_similarity_ties(self):
        # identical direction (sim ties); smaller byte_size should win on vpb
        passages = [_p("big", [1.0, 0.0], byte_size=1000), _p("small", [1.0, 0.0], byte_size=50)]
        ranked = rank_passages([1.0, 0.0], passages, k=2, value_per_byte=True)
        assert ranked[0].passage.passage_id == "small"

    def test_plain_similarity_does_not_depend_on_byte_size(self):
        passages = [_p("big", [1.0, 0.0], byte_size=1000), _p("small", [1.0, 0.0], byte_size=50)]
        ranked = rank_passages([1.0, 0.0], passages, k=2, value_per_byte=False)
        # tie on similarity -> deterministic tie-break by passage_id desc
        assert {r.passage.passage_id for r in ranked} == {"big", "small"}
        assert all(math.isclose(r.similarity, 1.0, abs_tol=1e-6) for r in ranked)
