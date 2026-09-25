"""Idea 2: carbon-intensity-aware scheduling for green 5G retrieval.

The energy model (ml_retriever.energy) turns a query into joules; joules become
gCO2e via the grid's carbon intensity. But grid intensity is NOT constant -- it
swings ~2-3x over a day as the generation mix changes (wind/solar vs gas/coal).
So the SAME query costs very different carbon depending on WHEN it runs.

This module adds the missing time axis:

  * `CarbonIntensityModel` -- a 24-hour diurnal gCO2e/kWh trace with the
    canonical shape of a solar-heavy grid: a midday trough (solar floods the
    grid) and a sharp evening peak (solar drops, demand stays high, gas ramps).
    Values are a documented, swappable stand-in for a real signal (e.g. an
    Electricity Maps / WattTime feed) -- the SHAPE is what the scheduling result
    depends on, and it matches published daily curves.

  * `carbon_aware_budget` -- scales a retrieval byte/energy budget by how clean
    the grid currently is, so heavy retrieval leans into green windows and pulls
    back when the grid is dirty (a deadline-free, latency-tolerant agent can do
    this without hurting the user).

  * `defer_savings` -- for a latency-tolerant query that may run any time within
    a `window_h`-hour slack, the gCO2e saved by shifting it to the greenest hour
    in that window versus running it at arrival.

Pure/deterministic, so it is unit-tested directly with no model or network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A representative solar-heavy daily carbon-intensity curve, gCO2e/kWh, one value
# per local hour 0..23. Overnight moderate (wind, baseload), a deep midday solar
# trough (~10:00-15:00), and a high evening ramp/peak (~18:00-21:00). Mean ~475
# gCO2e/kWh (IEA global-average, matching energy.py's default) so the average
# case reconciles with the static accounting; only the WITHIN-DAY spread drives
# the scheduling result. Shape follows published grid curves (e.g. CAISO/UK).
DEFAULT_HOURLY_GCO2E_PER_KWH = [
    430, 410, 395, 385, 380, 400,   # 00-05  overnight (wind/baseload)
    460, 520, 560, 540, 470, 400,   # 06-11  morning ramp, then solar rising
    330, 300, 310, 360, 450, 560,   # 12-17  midday solar trough -> late-day ramp
    650, 690, 640, 560, 500, 460,   # 18-23  evening peak, then decline
]


@dataclass
class CarbonIntensityModel:
    """Hourly grid carbon intensity (gCO2e/kWh) as a function of local hour.

    `hourly` is a 24-length list; `intensity(hour)` reads it (fractional hours
    are floored to the hour). All scheduling logic below is expressed against
    this so a reviewer can drop in a real regional feed unchanged."""
    hourly: list = field(default_factory=lambda: list(DEFAULT_HOURLY_GCO2E_PER_KWH))

    def __post_init__(self):
        if len(self.hourly) != 24:
            raise ValueError("hourly must have exactly 24 values (one per hour)")

    def intensity(self, hour: float) -> float:
        return float(self.hourly[int(hour) % 24])

    @property
    def mean(self) -> float:
        return sum(self.hourly) / 24.0

    @property
    def greenest_hour(self) -> int:
        return min(range(24), key=lambda h: self.hourly[h])

    @property
    def dirtiest_hour(self) -> int:
        return max(range(24), key=lambda h: self.hourly[h])

    def greenest_hour_in_window(self, start_hour: float, window_h: int) -> int:
        """The hour in [start_hour, start_hour+window_h) with lowest intensity
        (wrapping past midnight)."""
        start = int(start_hour) % 24
        span = [(start + d) % 24 for d in range(max(window_h, 1))]
        return min(span, key=lambda h: self.hourly[h])


def gco2e(energy_j: float, intensity_g_per_kwh: float) -> float:
    """Joules -> gCO2e at a given grid intensity (J -> kWh -> gCO2e)."""
    return (energy_j / 3.6e6) * intensity_g_per_kwh


def carbon_aware_budget(base_budget: float, model: CarbonIntensityModel, hour: float,
                        min_scale: float = 0.5, max_scale: float = 1.5) -> float:
    """Scale a retrieval budget (bytes or joules) by how clean the grid is now.

    scale = mean_intensity / current_intensity, clamped to [min_scale, max_scale]:
    a cleaner-than-average grid (scale>1) permits a larger budget, a dirtier grid
    (scale<1) shrinks it. Clamped so scheduling never starves or explodes the
    budget. Returns the adjusted budget in the same unit as `base_budget`."""
    scale = model.mean / max(model.intensity(hour), 1e-9)
    scale = max(min_scale, min(max_scale, scale))
    return base_budget * scale


def defer_savings(energy_j: float, model: CarbonIntensityModel, arrival_hour: float,
                  window_h: int = 6) -> dict:
    """For a latency-tolerant query of `energy_j` joules arriving at
    `arrival_hour` that may run any time within the next `window_h` hours, the
    gCO2e of running it now vs shifting it to the greenest hour in the window."""
    at_arrival = gco2e(energy_j, model.intensity(arrival_hour))
    best_h = model.greenest_hour_in_window(arrival_hour, window_h)
    at_best = gco2e(energy_j, model.intensity(best_h))
    saved = at_arrival - at_best
    return {
        "arrival_hour": int(arrival_hour) % 24,
        "run_hour": best_h,
        "gco2e_at_arrival": at_arrival,
        "gco2e_deferred": at_best,
        "gco2e_saved": saved,
        "pct_saved": (100.0 * saved / at_arrival) if at_arrival else 0.0,
    }
