"""Tests for Idea 2: carbon-intensity-aware scheduling (ml_retriever.carbon)."""
import pytest

from ml_retriever.carbon import (
    CarbonIntensityModel, carbon_aware_budget, defer_savings, gco2e,
)


def test_trace_has_24_hours_and_diurnal_shape():
    m = CarbonIntensityModel()
    assert len(m.hourly) == 24
    # midday trough is cleaner than the evening peak
    assert m.hourly[13] < m.hourly[19]
    assert m.greenest_hour in range(10, 16)   # solar trough
    assert m.dirtiest_hour in range(17, 22)    # evening peak


def test_bad_length_rejected():
    with pytest.raises(ValueError):
        CarbonIntensityModel(hourly=[400] * 23)


def test_gco2e_conversion():
    # 3.6e6 J = 1 kWh, so at 475 gCO2e/kWh one kWh emits 475 g.
    assert gco2e(3.6e6, 475.0) == pytest.approx(475.0)


def test_carbon_aware_budget_expands_when_clean_shrinks_when_dirty():
    m = CarbonIntensityModel()
    green = carbon_aware_budget(1000.0, m, m.greenest_hour)
    dirty = carbon_aware_budget(1000.0, m, m.dirtiest_hour)
    assert green > 1000.0 >= dirty  # clean grid -> bigger budget, dirty -> smaller
    # clamped to the configured band
    assert green <= 1500.0 + 1e-6 and dirty >= 500.0 - 1e-6


def test_defer_saves_carbon_when_greener_hour_exists():
    m = CarbonIntensityModel()
    # arriving at the dirtiest hour with slack should let us shift to a cleaner one
    r = defer_savings(100.0, m, arrival_hour=m.dirtiest_hour, window_h=12)
    assert r["gco2e_saved"] > 0
    assert r["pct_saved"] > 0
    assert m.hourly[r["run_hour"]] <= m.hourly[r["arrival_hour"]]


def test_defer_zero_when_already_greenest():
    m = CarbonIntensityModel()
    r = defer_savings(100.0, m, arrival_hour=m.greenest_hour, window_h=1)
    assert r["gco2e_saved"] == pytest.approx(0.0)
