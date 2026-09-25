# Carbon-intensity-aware scheduling (Idea 2)

Per-query 5G energy (policy `linucb_energy`, html_page transfer+compute + RRC radio fast-dormancy): **141.8 J**.

Grid carbon trace: mean 465 gCO2e/kWh, min 300 (hr 13), max 690 (hr 19); within-day spread 2.30x.

| slack window | baseline mgCO2e/day | scheduled mgCO2e/day | saved | % saved |
|---|---|---|---|---|
| 2 h | 439.44 | 417.00 | 22.44 | 5.1% |
| 4 h | 439.44 | 378.02 | 61.43 | 14.0% |
| 6 h | 439.44 | 350.85 | 88.60 | 20.2% |
| 12 h | 439.44 | 316.00 | 123.45 | 28.1% |
| 24 h | 439.44 | 283.51 | 155.93 | 35.5% |

Ceiling (run every query at the greenest hour): **35.5%** gCO2e reduction.

Workload: 24 queries, one per hour (uniform demand). The %% saving is independent of the absolute per-query energy (it cancels in the ratio) and comes purely from shifting flexible retrieval into greener grid windows -- a lever orthogonal to, and stacking on top of, the energy-aware stopping policy (Idea 1).
