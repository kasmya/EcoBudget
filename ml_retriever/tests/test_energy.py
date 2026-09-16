"""Phase A energy-accounting tests (pure math, deterministic)."""

import math

from ml_retriever.energy import (
    ComputeModel, EnergyAccountant, FiveGTransferModel, OpCounts, query_op_counts,
)


class TestTransfer:
    def test_scales_with_bytes(self):
        m = FiveGTransferModel()
        assert m.energy_joules(2000) == 2 * m.energy_joules(1000)

    def test_ran_dominates(self):
        m = FiveGTransferModel()
        b = m.breakdown_joules(10000)
        assert b["ran"] > b["transport"]  # RAN is the majority share

    def test_zero_bytes_zero_energy(self):
        assert FiveGTransferModel().energy_joules(0) == 0.0


class TestCompute:
    def test_scales_with_op_count(self):
        c = ComputeModel()
        one = c.energy_joules(OpCounts(qa_score=1))
        ten = c.energy_joules(OpCounts(qa_score=10))
        assert math.isclose(ten, 10 * one)

    def test_bigger_model_costs_more(self):
        c = ComputeModel()
        # answer_gen (flan-t5-base) vs query_embed (minilm), one call each
        assert c.energy_joules(OpCounts(answer_gen=1)) > c.energy_joules(OpCounts(query_embed=1))

    def test_more_efficient_device_uses_less(self):
        ops = OpCounts(qa_score=5, answer_gen=2)
        edge = ComputeModel(flops_per_joule=5e10).energy_joules(ops)
        npu = ComputeModel(flops_per_joule=5e12).energy_joules(ops)
        assert npu < edge


class TestAccountant:
    def test_total_is_compute_plus_transfer(self):
        a = EnergyAccountant()
        r = a.account(OpCounts(qa_score=3, answer_gen=2), num_bytes=1000)
        assert math.isclose(r["total_j"], r["compute_j"] + r["transfer_j"])
        assert 0.0 <= r["compute_fraction"] <= 1.0

    def test_gco2e_conversion(self):
        a = EnergyAccountant(grid_g_per_kwh=475.0)
        r = a.account(OpCounts(), num_bytes=0)  # all zero -> zero everything
        assert r["gco2e"] == 0.0


class TestOpCounts:
    def test_per_requirement_counts(self):
        # 2 requirements, 3 passages added, per-requirement answering
        ops = query_op_counts(n_requirements=2, n_passages_added=3, answer_mode="per_requirement")
        assert ops.decompose == 1
        assert ops.query_embed == 2
        assert ops.qa_score == 2 * 3
        assert ops.answer_gen == 2

    def test_joint_mode_one_generation(self):
        ops = query_op_counts(n_requirements=2, n_passages_added=3, answer_mode="joint")
        assert ops.answer_gen == 1


from ml_retriever.energy import PayloadModel
from ml_retriever.types import Passage


def _pg(pid, text_bytes, src):
    return Passage(passage_id=pid, text="x", byte_size=text_bytes, source_url=src)


class TestPayloadModel:
    P = [_pg("a", 200, "http://p1"), _pg("b", 100, "http://p1"), _pg("c", 150, "http://p2")]

    def test_text_is_sum_of_text_bytes(self):
        assert PayloadModel().transfer_bytes(self.P, "text") == 450

    def test_resource_inflates_with_floor(self):
        m = PayloadModel(resource_inflation=4.0, resource_floor_bytes=800)
        # 200*4=800, 100*4=400->floor 800, 150*4=600->floor 800
        assert m.transfer_bytes(self.P, "resource") == 800 + 800 + 800

    def test_html_page_dedups_by_source(self):
        m = PayloadModel(html_page_bytes=60_000)
        # two unique sources (p1, p2) -> 2 pages
        assert m.transfer_bytes(self.P, "html_page") == 2 * 60_000

    def test_full_page_much_larger_than_html(self):
        m = PayloadModel()
        assert m.transfer_bytes(self.P, "full_page") > 10 * m.transfer_bytes(self.P, "html_page")

    def test_crossover_transfer_can_dominate_at_page_scale(self):
        from ml_retriever.energy import EnergyAccountant
        a = EnergyAccountant()
        ops = OpCounts(qa_score=4, answer_gen=2)  # a few inferences
        text = a.account_passages(ops, self.P, "text")
        page = a.account_passages(ops, self.P, "full_page")
        assert text["compute_fraction"] > 0.9      # compute dominates at snippet scale
        assert page["compute_fraction"] < 0.1       # transfer dominates at full-page scale
