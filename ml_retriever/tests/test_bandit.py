"""Model-free tests for the Phase 5 bandit core: features, reward,
normalizer, and the online policy's learn/select/save behavior."""

import numpy as np
import pytest

from ml_retriever.bandit import (
    ACTIONS,
    FEATURE_NAMES,
    N_FEATURES,
    RETRIEVE,
    STOP,
    BanditPolicy,
    FeatureNormalizer,
    compute_reward,
    featurize,
)


class TestFeaturize:
    def test_vector_length_and_order(self):
        v = featurize({"frac_satisfied": 0.5, "step": 3})
        assert v.shape == (N_FEATURES,)
        assert v[FEATURE_NAMES.index("frac_satisfied")] == 0.5
        assert v[FEATURE_NAMES.index("step")] == 3.0

    def test_missing_features_default_zero(self):
        assert featurize({}).tolist() == [0.0] * N_FEATURES


class TestReward:
    def test_success_minus_byte_penalty(self):
        # success=1, used half the payload, lam=0.5 -> 1 - 0.5*0.5 = 0.75
        assert compute_reward(True, 50, 100, lam=0.5) == pytest.approx(0.75)

    def test_failure_is_negative_penalty(self):
        assert compute_reward(False, 100, 100, lam=0.5) == pytest.approx(-0.5)

    def test_more_bytes_lowers_reward(self):
        assert compute_reward(True, 20, 100, 0.5) > compute_reward(True, 80, 100, 0.5)


class TestNormalizer:
    def test_zscore(self):
        ctxs = [featurize({"step": 0}), featurize({"step": 2}), featurize({"step": 4})]
        norm = FeatureNormalizer().fit(ctxs)
        out = norm.transform(featurize({"step": 2}))  # the mean -> 0
        assert out[FEATURE_NAMES.index("step")] == pytest.approx(0.0)

    def test_constant_feature_does_not_divide_by_zero(self):
        ctxs = [featurize({"coverage": 1.0}) for _ in range(3)]
        norm = FeatureNormalizer().fit(ctxs)
        assert np.isfinite(norm.transform(featurize({"coverage": 1.0}))).all()


class TestBanditPolicy:
    def test_select_action_returns_valid_action(self):
        p = BanditPolicy(seed=1)
        a = p.select_action(featurize({"step": 0}))
        assert a in range(len(ACTIONS))

    def test_cold_start_values_are_neutral(self):
        p = BanditPolicy(seed=1)
        assert p.action_values(featurize({"step": 0})) == [0.5, 0.5]

    def test_learns_a_separable_signal(self):
        # context with has_unretrieved_candidate=1 -> RETRIEVE good (reward 1);
        # context with it =0 -> STOP good. Policy should learn to split them.
        p = BanditPolicy(epsilon=0.0, seed=3)
        ctx_more = featurize({"has_unretrieved_candidate": 1.0, "frac_remaining": 1.0})
        ctx_done = featurize({"has_unretrieved_candidate": 0.0, "frac_remaining": 0.0})
        for _ in range(60):
            p.update(ctx_more, RETRIEVE, reward=1.0)
            p.update(ctx_more, STOP, reward=0.0)
            p.update(ctx_done, STOP, reward=1.0)
            p.update(ctx_done, RETRIEVE, reward=0.0)
        assert p.select_action(ctx_more, explore=False) == RETRIEVE
        assert p.select_action(ctx_done, explore=False) == STOP

    def test_save_and_load_roundtrip(self, tmp_path):
        p = BanditPolicy(epsilon=0.0, seed=5)
        ctx = featurize({"has_unretrieved_candidate": 1.0})
        for _ in range(20):
            p.update(ctx, RETRIEVE, 1.0)
            p.update(ctx, STOP, 0.0)
        path = tmp_path / "policy.joblib"
        p.save(path)
        loaded = BanditPolicy.load(path)
        assert loaded.action_values(ctx) == p.action_values(ctx)
        assert loaded.select_action(ctx, explore=False) == RETRIEVE
