# EcoBudget

**Task-sufficient web retrieval for greener 5G question answering.**

EcoBudget is a research project that asks a simple question when a system answers
a question from the web, how little does it actually need to load? Most
retrieval-augmented systems either pull a fixed amount of context or grab whole
pages and let the model sort it out. Both move far more data than the answer
requires. On a 5G link that waste is not free, because moving bytes costs radio
energy and carbon, and every fetch keeps the device modem in a high-power state.
EcoBudget decides, per question, exactly how much evidence it needs, stops as soon
as it has enough, and then measures the bytes and energy it saved.

Everything runs on local models. There is no hosted LLM API anywhere in the
pipeline, and that rule is enforced in code, so the efficiency and energy numbers
reflect the system we actually run rather than a service we call.

## What is in this repository

EcoBudget has three workstreams:

- **`ml_retriever/`** is the reproducible machine-learning system and the heart of
  the project: task decomposition, entity-aware retrieval, evidence tracking, the
  adaptive stopping policy, answer generation, the 5G energy and radio model, a
  verdict-aware evaluation, an energy-aware stopping reward, a
  carbon-intensity-aware scheduler, and the full experiment harness. It has 189
  passing tests and a detailed README of its own. Start there:
  [`ml_retriever/README.md`](ml_retriever/README.md).
- **`backend/`** is an earlier FastAPI prototype that fetches real web pages with
  Playwright, extracts passages, and estimates transfer CO2e. We reuse its real
  fetched pages (`backend/pages/`) to ground the energy model in measured page
  sizes, and we drive the end-to-end demo against live pages the same way.
- **`index.html`** (repo root) is the project landing page, a premium single
  page with a full-screen hero video and an interactive iPhone demo that animates
  the task-sufficient loading idea using our measured numbers. Its assets live in
  **`frontend/`** (`frontend/public/`, reachable from root via the `public`
  symlink) alongside the reusable UI components.
- **`dashboard.html`** (repo root) is a live EcoBudget-versus-normal demo. Served
  by `backend/demo_server.py`, it calls a live web-search API server-side and
  compares loading every full page (normal) against loading one task-sufficient
  snippet (EcoBudget), showing real byte and 5G energy savings across payload
  scales. See "The live demo" below.
- **`ecobudget_app/`** is a separate carbon-budget application built alongside the
  research system (see `ecobudget_app/README.md`).

## The idea in plain terms

We treat retrieval as a task-sufficiency problem. Rather than asking "can the
system answer," we ask "what is the smallest amount of loading that still answers
correctly," and then we check whether the energy saved by loading less actually
beats the energy spent running the models. The pipeline does four things:

1. **Decompose.** Break a question into small, explicit information needs, each a
   structured (entity, attribute) pair, for example (iPhone 16, price).
2. **Retrieve.** For each requirement, gate candidate passages to the right entity
   first, then rank by the attribute so the right fact is surfaced.
3. **Decide.** A learned stopping policy chooses STOP or RETRIEVE after each step,
   trading marginal evidence coverage against byte cost.
4. **Answer and measure.** Generate the answer from only the evidence gathered,
   then record the bytes moved and the energy that implies on a 5G access path.

## Architecture

The pipeline is a chain of small, independent, dependency-injected components,
each unit tested without downloading a model.

| stage | component | file |
| --- | --- | --- |
| Decompose | flan-t5-base + LoRA, snapped to a closed 35-attribute vocabulary | `ml_retriever/decomposer.py` |
| Retrieve | MiniLM bi-encoder with an entity-aware two-stage retriever | `ml_retriever/retriever.py` |
| Track evidence | RoBERTa question-answering confidence, not raw similarity | `ml_retriever/evidence.py` |
| Decide (stop) | LinUCB (default), plus an SGD bandit and Thompson sampling | `ml_retriever/bandit.py`, `rollout.py` |
| Answer | flan-t5-base + LoRA, one call per requirement, with abstain | `ml_retriever/answer.py` |
| Judge | answer-type-aware checker, reports judge_success and fact_f1 | `ml_retriever/judge.py` |
| Energy | compute FLOPs, 5G per-bit transfer, payloads, RRC radio model | `ml_retriever/energy.py` |
| Orchestrate | the end-to-end deployed path | `ml_retriever/system.py` |

## Key results

All values are measured by us on the frozen test set or on real pages, unless a
line says validation. Full detail and confidence intervals are in
[`ml_retriever/docs/phase_g_consolidation.md`](ml_retriever/docs/phase_g_consolidation.md).

- **Retrieval:** entity-aware retrieval raised recall@1 from 0.914 to 0.989 and
  recall@5 from 0.983 to a perfect 1.000 on 175 unique requirements.
- **Decomposition:** exact-match 0.892 on validation after the Phase C retrain.
- **Answering with far fewer bytes:** on the frozen test set the adaptive methods
  reach 0.895 fact-coverage success and 0.944 fact_f1 at about 106 bytes per
  query, versus 420 bytes for full-page loading (about 75 percent fewer bytes) at
  statistically equal success.
- **Verdict-aware success (the honest headline):** the fact-coverage metric passes
  as long as the required value is present, which is lenient for yes/no and
  comparison questions. Scoring the actual verdict (the correct yes/no, and which
  item is bigger) with `verdict.py` gives 0.859 overall; yes_no is genuinely 64/64
  = 1.000, single_fact 1.000, multi_part 0.500, numeric-which comparisons 1/3, and
  procedure/narrative 0.000 (no training coverage). Report 0.859, not 0.895, as
  the success number.
- **Energy at realistic page scale:** using the measured median page (506 KB),
  adaptive stopping saved about 66 joules per query versus full-page loading on
  the frozen test set (95 percent CI [41, 95]); transfer was about 95 percent of
  total energy at that scale.
- **5G radio energy:** fewer, earlier fetches cut radio and tail energy. Under
  fast-dormancy release the policy uses about 4 times less radio energy than full
  retrieval (about 26 J versus about 107 J), robust across the cited-coefficient
  sensitivity sweep.
- **Real live web pages:** running the pipeline end to end against a live web
  search across domains, we measured about 92 to 99 percent fewer bytes and about
  42 joules of radio energy saved per query.
- **Energy-aware stopping (research extension, Idea 1):** an energy-aware reward
  (`reward_mode="energy"`, tuned mu=0.15) holds frozen-test success 0.895 at about
  25.9 J radio, matching the deployed byte-reward LinUCB (26.0 J) and about 4 times
  below full retrieval (106.7 J). This confirms the deployed policy is already near
  energy-optimal. See `ml_retriever/docs/phase_g_consolidation.md`.
- **Carbon-intensity-aware scheduling (research extension, Idea 2):** shifting
  latency-tolerant retrieval into the greenest hour of a slack window cuts gCO2e by
  about 5, 14, 20, and 28 percent at 2, 4, 6, and 12 hour windows (ceiling about 36
  percent), and stacks on Idea 1. See `ml_retriever/docs/carbon_schedule.md`.
- **An honest finding:** with a near-perfect retriever the learned bandit
  converges to a strong hand-tuned heuristic, and across five seeds our SGD bandit
  is unstable (0.790 plus or minus 0.283) while LinUCB is stable (0.913 plus or
  minus 0.007). We therefore deploy LinUCB and frame the contribution as the
  task-sufficiency system and 5G energy characterisation, not a new bandit.

## Repository structure

```
ecobudget/
  README.md                     this file
  index.html                    landing page (served at repo root)
  dashboard.html                live EcoBudget-vs-normal demo page
  plan.md                       the original ML workstream plan
  codebase-analysis.md          walkthrough of the backend prototype
  researchposter.md             conference poster content
  extended_abstract.md          extended abstract draft
  poster_abstract.md            poster abstract; poster_pptx_content.md poster text
  ml_retriever/                 the ML system (see its own README)
    ml_retriever/               the package (decomposer, retriever, energy,
                                verdict, carbon, ...)
    data/                       corpus, tasks, frozen splits, measured payloads
    scripts/                    build, train, evaluate, figures, verdict/carbon evals
    docs/                       roadmap, energy model, eval notes, verdict eval,
                                carbon schedule, diagnostics, reproducibility
    figures/                    the generated paper figures
    tests/                      189 tests
  backend/                      FastAPI prototype, real page fetching, demo_server
  frontend/                     landing-page assets and reusable UI components
  ecobudget_app/                separate carbon-budget application
```

## Getting started

### The ML system (recommended entry point)

```bash
cd ml_retriever
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
python -m pytest -q                  # expect 189 passing, seconds, no downloads
```

Run the working pipeline on a few tasks (uses the trained checkpoints; LinUCB is
the default stopping policy):

```bash
python scripts/run_pipeline.py --n 20 --policy linucb \
    --answerer models/answer-base --answer_mode per_requirement
```

Reproduce the headline experiments, the figures, and the frozen test run: see the
step-by-step commands in [`ml_retriever/README.md`](ml_retriever/README.md) and
the exact environment, seeds, and per-number provenance in
[`ml_retriever/docs/reproducibility.md`](ml_retriever/docs/reproducibility.md).

### The landing page

```bash
python3 -m http.server 8000
# then open http://localhost:8000/  (index.html is served by default)
```

### The live demo

`backend/demo_server.py` is a small hardened static server that also exposes a
`/api/compare` endpoint. It calls a live web-search API server-side and compares
loading every full page (normal) against loading one task-sufficient snippet
(EcoBudget), computing real byte and 5G energy savings with `ml_retriever.energy`.

```bash
ml_retriever/.venv/bin/python backend/demo_server.py
# then open http://localhost:8770/ and follow "Live demo" to dashboard.html
```

The server auto-loads a web-search API key from a gitignored `.env` at the repo
root. You must provide your own key; never commit it.

### The backend prototype (optional)

The backend has no dependency manifest yet; install its imports manually (FastAPI,
BeautifulSoup, sentence-transformers, Playwright) as listed in
[`codebase-analysis.md`](codebase-analysis.md), then:

```bash
cd backend
uvicorn main:app --reload
```

Fixture pages under `backend/pages/` are not tracked in git; regenerate them with
`backend/download_pages.sh` or `backend/fetch_missing_pages.py`.

## Local models only

The `ml_retriever` package must never depend on a hosted LLM API. A guard scans
for banned imports and known API hosts, and a test runs it as part of the suite:

```bash
cd ml_retriever && python scripts/check_no_llm_api.py
```

## Documentation

- [`ml_retriever/README.md`](ml_retriever/README.md): full system README with
  architecture, datasets, training, and reproduction.
- [`ml_retriever/docs/roadmap_5g_green.md`](ml_retriever/docs/roadmap_5g_green.md):
  the phased plan (A through H) with acceptance gates.
- [`ml_retriever/docs/energy_model.md`](ml_retriever/docs/energy_model.md): every
  energy and radio coefficient, its unit, and its source.
- [`ml_retriever/docs/eval_notes.md`](ml_retriever/docs/eval_notes.md): per-phase
  measured results and the decisions behind them.
- [`ml_retriever/docs/phase_g_consolidation.md`](ml_retriever/docs/phase_g_consolidation.md):
  consolidated ablations, multi-seed variance, energy-aware stopping, and the
  final test run.
- [`ml_retriever/docs/verdict_eval.md`](ml_retriever/docs/verdict_eval.md): the
  verdict-aware evaluation (yes/no and which-is-bigger), why it is stricter than
  fact-coverage, and the 0.859 result.
- [`ml_retriever/docs/carbon_schedule.md`](ml_retriever/docs/carbon_schedule.md):
  carbon-intensity-aware scheduling (Idea 2), the diurnal trace, and gCO2e savings.
- [`ml_retriever/docs/diagnostics_report.md`](ml_retriever/docs/diagnostics_report.md):
  overfitting, leakage, calibration, latency, and subgroup diagnostics.
- [`ml_retriever/docs/reproducibility.md`](ml_retriever/docs/reproducibility.md):
  environment, seeds, checkpoints, and per-number provenance.
- [`ml_retriever/docs/DEVELOPER.md`](ml_retriever/docs/DEVELOPER.md): developer
  orientation to the package and scripts.
- [`researchposter.md`](researchposter.md) and [`extended_abstract.md`](extended_abstract.md):
  the writeup material.

## Project status

Research phases A through G are complete: compute-versus-transfer accounting, a
5G-grounded energy model on measured payloads, dataset scale-up, the entity-aware
retriever, external baselines, the radio-state model, and the consolidation phase
(ablations, multi-seed, frozen test run, reproducibility appendix). Two research
extensions are also implemented and evaluated: an energy-aware stopping reward
(Idea 1) and carbon-intensity-aware scheduling (Idea 2). A verdict-aware
evaluation and a diagnostics suite (overfitting, leakage, calibration, latency,
subgroups) were added for rigor. Phase H, the paper writeup, is in progress.

## Limitations

We state these plainly so nothing is oversold.

- The contribution is the task-sufficiency system and the 5G energy and radio
  characterisation, not a new learning algorithm. LinUCB is the recommended and
  deployed learned policy; the custom SGD bandit is high-variance and is kept as
  an ablation.
- Answer quality degrades on raw live-page text (a corpus-to-web domain gap). The
  byte and energy savings transfer to the real web, but real-web answer accuracy
  needs per-entity fetching with structured extraction, which is future work.
- The corpus is domain-narrow and electronics-skewed (phones and laptops are about
  half of the tasks), and some source URLs are synthetic, so external validity
  beyond these domains is untested. The clean in-domain corpus is part of why the
  numbers are high. Weak subgroups are reported honestly: multi_part about 0.50,
  and procedure and narrative fail outright for lack of training coverage.
- Fact-coverage success (0.895) is lenient for yes/no and comparison questions; we
  therefore also report verdict-aware success (0.859) as the honest headline.
- Energy is a first-order model with cited coefficients and reported sensitivity
  ranges, not a hardware power measurement.

## License

This project is licensed under the Apache License 2.0. See the
[LICENSE](LICENSE) file for the full text.
