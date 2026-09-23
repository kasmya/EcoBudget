"""Idea 2 evaluation: carbon-intensity-aware scheduling of green 5G retrieval.

Takes the deployed policy's measured per-query 5G energy (from phase7_results;
transfer+compute at the realistic html_page scale plus RRC radio, fast-dormancy)
and the diurnal carbon-intensity trace (ml_retriever.carbon), and reports the
gCO2e a latency-tolerant workload saves by shifting each query to the greenest
hour within a slack window, versus running at arrival time.

The % saving depends only on the carbon trace and the slack window (query energy
cancels in the ratio), so it is deterministic; the absolute mgCO2e uses the
measured mean energy per query. Writes docs/carbon_schedule.md +
data/carbon_schedule_results.json.

Usage: python scripts/carbon_schedule_eval.py
"""
import json
from pathlib import Path

from ml_retriever.carbon import CarbonIntensityModel, defer_savings, gco2e

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
POLICY = "linucb_energy"   # the energy-aware policy (Idea 1); falls back to bandit
WINDOWS = [2, 4, 6, 12, 24]


def per_query_energy_j(results: dict, cond: str) -> float:
    a = results.get(cond) or results["bandit"]
    # full 5G energy per query: transfer+compute at html_page scale + RRC radio
    # (fast-dormancy, the honest per-fetch-wake regime).
    return a["total_j_html_page"] + a["radio_j_fastdormancy"]


def main():
    results = json.loads((DATA / "phase7_results.json").read_text())
    energy_j = per_query_energy_j(results, POLICY)
    model = CarbonIntensityModel()

    # Workload: one query arriving each hour of the day (uniform 24h demand).
    arrivals = list(range(24))
    baseline_g = sum(gco2e(energy_j, model.intensity(h)) for h in arrivals)

    rows = []
    for w in WINDOWS:
        deferred_g = 0.0
        per_hour = []
        for h in arrivals:
            r = defer_savings(energy_j, model, arrival_hour=h, window_h=w)
            deferred_g += r["gco2e_deferred"]
            per_hour.append(r)
        saved = baseline_g - deferred_g
        rows.append({
            "window_h": w,
            "gco2e_baseline_mg": baseline_g * 1000,
            "gco2e_scheduled_mg": deferred_g * 1000,
            "gco2e_saved_mg": saved * 1000,
            "pct_saved": 100 * saved / baseline_g if baseline_g else 0.0,
        })

    # best single fixed green window (always run at the greenest hour) as the ceiling
    ceiling_g = 24 * gco2e(energy_j, model.intensity(model.greenest_hour))
    ceiling_pct = 100 * (baseline_g - ceiling_g) / baseline_g

    out = {
        "policy": POLICY, "energy_j_per_query": energy_j,
        "carbon_mean_g_per_kwh": model.mean,
        "carbon_min_g_per_kwh": model.hourly[model.greenest_hour],
        "carbon_max_g_per_kwh": model.hourly[model.dirtiest_hour],
        "carbon_spread_x": model.hourly[model.dirtiest_hour] / model.hourly[model.greenest_hour],
        "greenest_hour": model.greenest_hour, "dirtiest_hour": model.dirtiest_hour,
        "windows": rows,
        "ceiling_pct_saved": ceiling_pct,
    }
    (DATA / "carbon_schedule_results.json").write_text(json.dumps(out, indent=2))

    md = "# Carbon-intensity-aware scheduling (Idea 2)\n\n"
    md += (f"Per-query 5G energy (policy `{POLICY}`, html_page transfer+compute + "
           f"RRC radio fast-dormancy): **{energy_j:.1f} J**.\n\n")
    md += (f"Grid carbon trace: mean {model.mean:.0f} gCO2e/kWh, "
           f"min {out['carbon_min_g_per_kwh']:.0f} (hr {model.greenest_hour}), "
           f"max {out['carbon_max_g_per_kwh']:.0f} (hr {model.dirtiest_hour}); "
           f"within-day spread {out['carbon_spread_x']:.2f}x.\n\n")
    md += "| slack window | baseline mgCO2e/day | scheduled mgCO2e/day | saved | % saved |\n"
    md += "|---|---|---|---|---|\n"
    for r in rows:
        md += (f"| {r['window_h']} h | {r['gco2e_baseline_mg']:.2f} | "
               f"{r['gco2e_scheduled_mg']:.2f} | {r['gco2e_saved_mg']:.2f} | "
               f"{r['pct_saved']:.1f}% |\n")
    md += (f"\nCeiling (run every query at the greenest hour): "
           f"**{ceiling_pct:.1f}%** gCO2e reduction.\n\n")
    md += ("Workload: 24 queries, one per hour (uniform demand). The %% saving is "
           "independent of the absolute per-query energy (it cancels in the ratio) "
           "and comes purely from shifting flexible retrieval into greener grid "
           "windows -- a lever orthogonal to, and stacking on top of, the "
           "energy-aware stopping policy (Idea 1).\n")
    (DOCS / "carbon_schedule.md").write_text(md)

    print(f"per-query energy = {energy_j:.1f} J; carbon spread {out['carbon_spread_x']:.2f}x")
    for r in rows:
        print(f"  window {r['window_h']:>2}h: {r['pct_saved']:>5.1f}% gCO2e saved "
              f"({r['gco2e_saved_mg']:.2f} mg/day)")
    print(f"  ceiling (greenest hour): {ceiling_pct:.1f}%")
    print(f"Wrote {DOCS/'carbon_schedule.md'} and {DATA/'carbon_schedule_results.json'}")


if __name__ == "__main__":
    main()
