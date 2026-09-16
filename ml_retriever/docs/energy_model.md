# Energy model (Phase A) — 5G transfer vs compute

First-order energy accounting for the pipeline. Every coefficient is an
explicit, cited *assumption* and a constructor argument (`ml_retriever/energy.py`)
so it can be swapped. The aim is a defensible order-of-magnitude COMPUTE-vs-
TRANSFER comparison and a 5G-grounded transfer term — not a hardware measurement
(that is later-phase work).

## Compute energy (running the models)
`inference FLOPs ≈ 2 · params · (input_tokens + output_tokens)` per forward pass
(standard 2×MAC approximation), summed over every model call in a query, divided
by a device efficiency `flops_per_joule`.

| model | params | role |
|---|---|---|
| flan-t5-base | 248 M | decomposer, answer generator |
| roberta-base-squad2 | 125 M | evidence QA scorer |
| MiniLM-L6 | 22.7 M | query embedding |

- **Device efficiency (dominant assumption):** default `5e10 FLOPS/J`
  (≈50 GFLOPS/W, edge-CPU class). Mobile-SoC NPUs reach 1–10 TOPS/W; a server
  GPU differs — swap per target device.
- Token counts per op are documented averages in `energy.py::OP_TOKENS`.
- Ops per query (caching-independent): 1 decompose, 1 query-embed per
  requirement, **1 QA score per (requirement × added passage)**, and 1 answer
  generation per requirement (per-requirement mode). QA scoring is the dominant
  compute term because it scales with passages retrieved — which is exactly what
  the stopping policy controls.

## Transfer energy (5G access path)
`energy/bit = network_j_per_bit + device_rx_j_per_bit`, with the network term
split RAN + transport (RAN dominates 5G energy, ~70–80%).

- `network_j_per_bit = 2.25e-5` J/bit — from a mid-range mobile-access figure of
  ~0.05 kWh/GB for the 5G era (literature spans ~0.01–0.1 kWh/GB);
  0.05 kWh/GB = 1.8e5 J / 8e9 bit.
- `ran_fraction = 0.75` (RAN share); `device_rx_j_per_bit = 4.0e-7` (smartphone
  modem receive, assumption).
- **Payload note:** Phase A applies this to the passage *text* bytes
  (~25–259 B). Real transfers are KB–MB pages; **Phase B** adds realistic
  `page_bytes`, which will raise the transfer term and re-test the finding below.

## Carbon
`gCO2e = (total_J / 3.6e6 J/kWh) · grid_g_per_kwh`, default grid intensity
`475 gCO2e/kWh` (global-average, IEA-class assumption; green-host variants lower).

## Phase A finding (val, headline tasks, n=28)
- **Compute dominates transfer by ~100×.** Per query: compute ≈ 5–11 J,
  transfer ≈ 0.03–0.15 J → compute is **98.6–99.4%** of total energy. At
  short-snippet payloads, *saving bytes alone saves almost nothing.* The naive
  "fewer bytes ⇒ greener" story is false at this scale — quantified here.
- **But the policy still wins on TOTAL energy**, because compute is dominated by
  QA-scoring inferences that scale with passages retrieved. Fewer retrievals ⇒
  fewer QA/answer inferences ⇒ less compute. The bandit's total energy is
  **significantly lower** than every non-degenerate baseline:
  - vs heuristic: **−0.82 J/query [95% CI −1.58, −0.23]** (CI excludes 0)
  - vs fixed-1000B/1500B and full: **−4.9 to −5.1 J/query** (large, significant)
- **Reframed claim (stronger and honest):** the energy benefit of adaptive
  stopping comes from *doing less computation* (fewer evidence-scoring
  inferences), not from moving fewer bytes. Compute-aware adaptive retrieval is
  the lever; the transfer term is negligible until payloads are realistic
  (Phase B).

## Threats to validity
- Compute energy is FLOPs-based with assumed device efficiency and token counts
  (first-order). The compute≫transfer *ratio* (~100×) is robust to reasonable
  coefficient choices; absolute joules are not precise.
- Transfer uses text bytes; the "compute dominates" conclusion is
  payload-scale-dependent and is re-tested at realistic page sizes in Phase B.
- Per-query gCO2e is tiny in isolation; it matters at population scale.

---

# Phase B — realistic payloads & the compute↔transfer crossover

Phase A charged transfer at extracted-text bytes (~25–259 B), an optimistic
lower bound. On a real 5G link you fetch a web resource/page (KB–MB) to obtain a
passage. `PayloadModel` (`energy.py`) charges transfer across four scenarios,
de-duplicating page fetches by `source_url`:

| scenario | per-passage/page bytes | source (assumption) |
|---|---|---|
| text | extracted text (25–259 B) | Phase A lower bound |
| resource | text ×4, floor 800 B | HTML markup + HTTP/TLS overhead |
| html_page | 60 KB / unique page | HTTP-Archive-class median HTML document |
| full_page | 2 MB / unique page | HTTP-Archive-class median full page weight |

## The crossover finding (val, headline, n=28) — total J/query
| condition | text | resource | html_page | full_page |
|---|---|---|---|---|
| bandit | 6.09 | 6.35 | 25.3 | 647 |
| heuristic | 6.91 | 7.26 | 31.2 | 818 |
| fixed-1500B / full | 11.2 | 12.1 | 69.6 | 1961 |

**Bandit transfer share:** text 0.7% → resource 4.7% → **html_page 76%** →
full_page 99%.

**The compute-vs-transfer balance flips with payload realism.** At snippet scale
compute dominates (Phase A). At realistic web-page scale (**html_page onward,
the real RAG-over-web regime**) **transfer dominates** — so loading fewer
pages/bytes is the primary energy lever after all. The Phase A "compute
dominates" result was an artifact of unrealistically small payloads; it holds
only in the text/resource regime.

## The bandit wins in BOTH regimes (paired, 95% CI)
| bandit vs | net J (text) | net J (html_page) |
|---|---|---|
| heuristic | −0.82 [−1.58, −0.23] | **−5.9 [−10.5, −2.2]** |
| full | −5.12 [−5.75, −4.49] | **−44.3 [−51.3, −37.7]** |
| fixed-1000B | −4.86 [−5.42, −4.29] | **−42.0 [−47.6, −36.4]** |

All CIs exclude 0. **At realistic payloads the bandit's energy advantage grows
~7× vs the heuristic and reaches ~44 J/query vs full-page loading.** The green
claim is strongest exactly where it matters (real page fetches), driven by
retrieving fewer pages; in the compute-bound snippet regime it still wins via
fewer inferences.

## Threats to validity (Phase B)
- Page counts assume ~1 unique page per passage (our corpus rarely shares a
  `source_url`); if multiple passages share a page, page-scenario differences
  shrink — a conservative assumption for the adaptive policy.
- `html_page`/`full_page` sizes (60 KB, 2 MB) are HTTP-Archive-class medians,
  documented and swappable; the crossover *location* (between resource and
  html_page) is robust, the exact joules are not.
- No page is actually fetched; a measured deployment is later-phase work.
