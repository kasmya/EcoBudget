# EcoBudget: task-sufficient retrieval for greener 5G question answering

This is my workstream (Teammate 1, the ML and retrieval brain) of the EcoBudget
project. My goal is a research contribution for a 5G-green venue: show that a
question answering system can decide, per question, exactly how much web evidence
it needs, stop as soon as it has enough, and by doing so move fewer bytes and
spend less energy over a 5G link without losing answer quality.

Everything here runs on local models only. There is no hosted LLM API anywhere in
the pipeline, and that rule is enforced in code (see the "Local models only"
section). All numbers in this README were produced on my own machine and can be
reproduced with the commands below.

## The idea in plain terms

Most retrieval-augmented systems either load a fixed amount of context or load
everything they can and let the model sort it out. Both waste data. My system
does something more careful:

1. It breaks a question into small, explicit information needs. I call each one a
   requirement, written as an (entity, attribute) pair, for example
   (iPhone 16, price).
2. For each requirement it retrieves candidate passages, gated to the right
   entity first and then ranked by the attribute.
3. A learned stopping policy decides after each retrieval whether it already has
   enough evidence (STOP) or should pull one more passage (RETRIEVE).
4. It generates the answer from only the evidence it actually gathered, and I
   measure the bytes moved and the energy that implies on a 5G access path.

The scientific question I care about is not "can it answer" but "what is the
smallest amount of loading that still answers correctly", and whether the energy
saved by loading less actually beats the energy spent running the models.

## What the system is made of

The pipeline is a chain of small, independent, dependency-injected components, so
each one is unit tested without downloading a model.

- Decomposer (`ml_retriever/decomposer.py`): a fine-tuned flan-t5-base with a
  LoRA adapter that turns a question into requirements. Its output is snapped to
  a closed vocabulary of 35 attributes, which is a principled form of constrained
  decoding for a fixed schema.
- Retriever (`ml_retriever/retriever.py`): a MiniLM bi-encoder over a
  pre-embedded corpus. The production retriever is now the entity-aware
  two-stage version I built in Phase D (gate to the entity, then rank by the
  attribute).
- Evidence tracker (`ml_retriever/evidence.py`): scores whether the gathered
  passages actually satisfy each requirement, using a RoBERTa question answering
  confidence signal rather than raw similarity, because similarity alone is
  fooled by passages that mention the entity but not the attribute.
- Stopping policy (`ml_retriever/bandit.py`): a per-action contextual bandit
  (linear SGD classifier) trained with a per-step reward that trades marginal
  evidence coverage against byte cost.
- Answer generator (`ml_retriever/answer.py`): a fine-tuned flan-t5-base with a
  LoRA adapter, called once per requirement, with an abstain path.
- Judge (`ml_retriever/judge.py`): an answer-type-aware checker with a symmetric
  formatting normalizer. It reports two metrics separately: judge_success
  (binary, every required fact matched) and fact_f1 (the fraction of required
  facts matched).
- Energy model (`ml_retriever/energy.py`): compute energy from model FLOPs, 5G
  transfer energy from a per-bit access-path model, realistic payload scenarios,
  and the RRC radio-state and tail energy model I added in Phase F.
- System orchestrator (`ml_retriever/system.py`): the end-to-end deployed path
  that wires all of the above together.

## What I have built so far, phase by phase

I planned the work as phases A through H in `docs/roadmap_5g_green.md`. Here is
the honest status.

### Phase A: compute versus transfer energy accounting (done)

I measured the energy the pipeline spends running models per query and compared
it to the transfer energy saved by loading fewer bytes. Finding: at short-snippet
payloads, compute dominates transfer by roughly 100 times, so saving bytes alone
saves almost nothing at that scale. The policy still wins on total energy, because
fewer retrievals means fewer evidence-scoring inferences. Documented in
`docs/energy_model.md`.

### Phase B: a defensible 5G energy model on realistic payloads (done)

I replaced a single generic web-CO2 number with a 5G per-bit model split into
radio access network, transport, and device modem receive, and applied it across
four payload realism scenarios (text, resource, html_page, full_page). Finding:
the compute-versus-transfer balance flips with payload realism. At realistic web
page scale (html_page onward) transfer dominates, so loading fewer pages becomes
the primary energy lever. The bandit wins in both regimes.

Tier A update: I later grounded the html_page scenario in REAL measured pages
instead of an assumed 60 KB. `scripts/measure_real_pages.py` measures the 19 real
pages the sibling `backend/` prototype fetched with Playwright: median rendered
HTML is 506 KB (my old assumption was 8 times too small). With the measured page
weight, the policy's energy advantage over full-load grows from about 19 to about
109 joules per query at page scale (confidence interval [74.7, 148.3], excluding
zero), and transfer rises to 95 percent of total energy. On the real pages,
task-sufficient extraction transfers roughly 130 bytes versus a 506 KB document,
a byte reduction above 99.9 percent. See `docs/energy_model.md` Tier A addendum.

### Phase C: dataset scale-up (done)

The old dataset was too small for trustworthy statistics. I scaled it up and
retrained all three models. Current dataset:

- Corpus: 186 passages (up from about 90), across phones, laptops, electronics,
  travel, buildings, geography, countries, running shoes, EVs, earbuds, and
  mountains.
- Seed tasks: 344 hand-authored, with curated required_facts taken straight from
  the specs so training targets match evaluation exactly.
- Total tasks: 1376 (seed plus synthetic paraphrases).
- Frozen splits: train 1008, val 240, test 128, grouped by seed so no paraphrase
  leaks across splits, with a coverage invariant enforced in code.

### Phase D: entity-aware retriever (done, headline win)

The old retriever suffered from entity dominance: the entity name dominated the
query embedding, so it would return the wrong near-twin (ask for Galaxy S24 and
get S24 Ultra), or the wrong attribute of the right entity (ask for Empire State
Building height and get its floor count). I first measured the failures, then
fixed them with a two-stage retriever: gate to the requirement's entity (exact
match on metadata offline, word-boundary text match as a live fallback, full
corpus fallback for unknown entities), then rank the gated pool by the attribute
phrase alone.

Result on 175 unique seed requirements:

| retriever | recall@1 | recall@5 |
| --- | --- | --- |
| whole-question baseline | 0.589 | 0.931 |
| per-requirement bi-encoder (old) | 0.914 | 0.983 |
| entity-aware, attribute-ranked (new) | 0.989 | 1.000 |

recall@5 reaching 1.000 means every list and amenities passage is now retrieved,
which was the known failure the roadmap flagged. Two residual rank-1 misses
remain, both genuine attribute near-synonyms with the gold at rank 2, and I left
them honest rather than hacking around them.

### Phase G consolidation (partly done, the part I scoped)

After wiring the entity-aware retriever into training and evaluation, I re-ran
Phase 7 on validation. This surfaced a real regression that I fixed the
disciplined way, and it changed the paper's story honestly:

- The entity gate shrank the per-query byte scale, so the bandit's old byte
  penalty (lambda = 4.0, tuned against the old retriever) over-penalized
  retrieval and the policy collapsed to always-STOP. I re-ran the pre-registered
  lambda sweep, found a sharp cliff, and selected lambda = 0.5 by the same rule
  as before. This is validation-only hyperparameter selection; the test split
  stayed frozen.
- Corrected Phase 7 (validation, 92 headline comparison and single_fact tasks,
  generative answerer):

| condition | success | fact_f1 | avg bytes |
| --- | --- | --- | --- |
| one_per_req | 0.946 | 0.973 | 124 |
| heuristic | 0.946 | 0.973 | 134 |
| bandit (lambda 0.5) | 0.946 | 0.973 | 134 |
| fixed-500B | 0.957 | 0.978 | 426 |
| full | 0.957 | 0.957 | 552 |

- The honest finding: under a near-perfect retriever the learned bandit converges
  to the hand-designed heuristic (identical policy, difference 0 with confidence
  interval [0, 0]), because with recall@1 at 0.989 there is little uncertainty
  left to exploit. The bandit still clearly dominates the fixed budgets and the
  full-load baseline (about 400 to 418 fewer bytes and 17.7 to 19.3 fewer joules
  per query at html_page scale, confidence intervals excluding zero, at
  statistically equal success). So the paper claim is that the learned policy is
  Pareto-optimal and dominates fixed budgets, and its edge over a strong
  heuristic grows with retrieval uncertainty. I do not claim it beats the
  heuristic under the entity-aware retriever.

### Phase F: radio-state and tail energy, plus the noise characterization (done)

This is the actual 5G novelty. 5G energy is dominated by keeping the modem in the
high-power RRC_CONNECTED state, not only by bytes moved. After the last fetch an
inactivity timer holds the radio active before it releases to idle, which is the
classic tail energy waste. I added `RadioStateModel` to `energy.py`: one
promotion per query, the radio held active across the retrieval loop, then a tail
at tail power, with a fast-dormancy sensitivity toggle where each fetch pays its
own promotion and tail. I also added modeled end-to-end latency (compute time
plus transfer time). Phase 7 now prints a radio table and a latency table and
adds radio joules with a confidence interval to the bandit-versus-baseline block.

Radio and latency results (validation, 92 headline tasks):

| condition | fetches | radio_J (tight loop) | radio_J (fast dormancy) | end-to-end latency s |
| --- | --- | --- | --- | --- |
| bandit | 1.98 | 10.33 | 19.94 | 1.51 |
| heuristic | 1.98 | 10.33 | 19.94 | 1.51 |
| fixed-1000B | 8.92 | 12.09 | 89.90 | 3.28 |
| full | 9.05 | 12.12 | 91.22 | 3.31 |

The 8 joule inactivity tail is paid once by anything that fetches at all, so it is
constant across conditions. The differentiator is the number of fetches. In the
conservative tight-loop model the bandit saves about 1.8 joules of radio energy
versus full and fixed budgets (confidence interval [-1.87, -1.70], excluding
zero). Under the fast-dormancy sensitivity, where each fetch pays its own
promotion and tail, the gap explodes to roughly 71 joules per query versus
full-load. The bandit is also about 2.2 times faster end-to-end, so there is no
latency-versus-energy tradeoff here: fewer fetches win on both. As with bytes, the
bandit ties the heuristic (both issue about two fetches) and dominates the fixed
and full baselines.

Alongside this I added a pre-registered retrieval-noise sweep
(`scripts/retrieval_noise_sweep.py`) that states its hypothesis before looking at
results: the bandit's advantage over the heuristic should grow as retrieval
uncertainty rises. It injects controlled noise into retrieval, trains a fresh
bandit per noise level, and compares against the heuristic and fixed baselines on
validation (test split untouched).

| noise | bandit success / bytes / fetches | heuristic success / bytes | bandit minus heuristic success |
| --- | --- | --- | --- |
| 0.00 | 0.917 / 105 / 1.75 | 0.917 / 105 | +0.000 |
| 0.10 | 0.833 / 108 / 1.83 | 0.833 / 108 | +0.000 |
| 0.20 | 0.767 / 121 / 1.97 | 0.767 / 121 | +0.000 |
| 0.40 | 0.633 / 146 / 2.50 | 0.617 / 142 | +0.017 |
| 0.60 | 0.483 / 163 / 2.90 | 0.467 / 159 | +0.017 |

The hypothesis is confirmed in direction but the effect is small and I will not
oversell it. As noise rises the bandit does adaptively retrieve more (1.75 up to
2.90 fetches) and pulls ahead of the heuristic, but only by about 1.7 percentage
points at high noise, because the heuristic is itself somewhat adaptive (it
retrieves until an evidence threshold, so it also responds to noise). The
cleanest contrast is against fixed policies rather than the heuristic. This is an
honest characterization result about when adaptivity helps, not a headline win.

### Phase E: external baselines, multi-seed, and the frozen test run (done)

I added the standard contextual-bandit baselines a reviewer expects: LinUCB and
linear Thompson sampling (`ml_retriever/bandit.py`, trained by
`scripts/train_baselines.py`), sharing the exact features, per-step reward,
lambda, and normalizer as my SGD bandit, plus an Adaptive-RAG complexity-routing
analog. I then ran multi-seed variance and the single frozen test-split run.

Two honest findings came out of this, and I report both plainly:

1. All adaptive methods converge. On validation, my SGD bandit, LinUCB, LinTS,
   and the heuristic all reach the same success at about the same bytes, and all
   dominate the fixed and full baselines. I cannot claim my bandit is a better
   algorithm than the standard ones. It ties them.
2. My SGD bandit is unstable across seeds. Over five seeds it averages 0.790
   success with a large 0.283 standard deviation (it collapses on some seeds),
   while LinUCB (0.913 plus or minus 0.007) and LinTS (0.897 plus or minus 0.014)
   are stable, and the deterministic heuristic (0.917) is the most reliable. The
   single-seed number I had been quoting was a favorable seed.

Because of this I recommend a framing pivot: LinUCB is the headline learned
policy (stable, principled, standard, matches the heuristic, dominates fixed
budgets), and the real contribution of the paper is the task-sufficient framing
plus the 5G radio and energy characterization, not a new bandit algorithm.

Frozen test-split results (single final run, 124 headline tasks):

| condition | success | fact_f1 | avg bytes | fetches |
| --- | --- | --- | --- | --- |
| bandit | 0.895 | 0.944 | 106 | 1.84 |
| linucb | 0.895 | 0.944 | 108 | 1.87 |
| heuristic | 0.895 | 0.944 | 136 | 2.29 |
| one_per_req | 0.831 | 0.911 | 101 | 1.74 |
| full | 0.903 | 0.919 | 420 | 7.56 |

On the frozen test set the learned policies save about 22 percent of bytes versus
the heuristic at identical success (confidence interval excluding zero), beat
naive one-per-requirement on success by 6.5 points, and dominate the fixed and
full baselines by about 300 bytes and 12 to 13 joules per query at realistic page
scale (all confidence intervals excluding zero).

### Tier A/B capstone: end-to-end on real live web pages (done)

`scripts/end_to_end_web.py` fetches a real page (browser user-agent, with a
fixture fallback), splits its visible text into passages, and runs the full
pipeline over them, measuring bytes and energy against loading the whole page.
Measured on live gsmarena pages:

| question | full page | task-sufficient | byte reduction |
| --- | --- | --- | --- |
| iPhone 16 refresh rate | 54,012 B | 3,878 B | 92.8% |
| S24 Ultra weight | 56,296 B | 1,228 B | 97.8% |

Two honest takeaways. First, the core thesis holds on real web content: about 93
to 98 percent byte reduction and roughly 14 times less transfer energy, measured
on live pages, not modelled. Second, answer quality degrades on raw live-page
text, because the models were trained on a clean structured corpus and real pages
are messy spec dumps (the correct value is usually in the retrieved passage, but
the answerer formats it poorly). Closing that corpus-to-web domain gap, by
training on real-page chunks or adding a cleaner extraction stage, is the honest
next step for the answer side; the byte and energy savings already transfer to
the real web.

### Radio coefficients grounded in literature (Tier B, done)

The RRC radio-state and tail-energy coefficients are grounded in 5G/LTE
measurement studies (Narayanan et al., SIGCOMM 2021; Huang et al., MobiSys 2012)
and 3GPP TS 38.331, each with a plausible range rather than a single guessed
value. `scripts/radio_sensitivity.py` sweeps the ranges and confirms the adaptive
policy uses less radio energy than full-load in every one of 18 coefficient cells
(the per-query saving ranges from about 2 joules in the conservative tight-loop
model to about 100 joules under aggressive fast-dormancy release). See
`docs/energy_model.md` Tier B addendum for the source table.

### Phase H (not started)

- Phase H: the paper writeup itself (related work, figures, limitations).

## Key results at a glance

- Retrieval: recall@1 from 0.914 to 0.989, recall@5 from 0.983 to 1.000.
- Answering: 0.946 success and 0.973 fact_f1 on headline validation tasks.
- Efficiency: the policy matches the best success while using roughly 75 percent
  fewer bytes than fixed and full-load baselines.
- Energy: compute dominates at snippet scale, transfer dominates at realistic
  page scale (grounded in real measured pages: median HTML 506 KB), and the
  policy wins in both regimes, saving about 109 joules per query versus full-load
  at real page scale.
- Honesty: a strong retriever shrinks the learned policy's edge over a hand-tuned
  heuristic to zero; the value of adaptivity is a function of retrieval
  uncertainty, and I report that rather than hide it.
- Tests: 169 passing, and the no-LLM-API guard is green.

## Repository layout

```
ml_retriever/
  pyproject.toml
  ml_retriever/               the package
    types.py                  Requirement, Passage, AnswerResult
    interfaces.py             tracker and generator protocols
    decomposer.py             question to requirements (flan-t5 + LoRA)
    retriever.py              bi-encoder + EntityAwareRetriever (Phase D)
    evidence.py               QA-confidence evidence coverage
    bandit.py                 contextual bandit stopping policy
    rollout.py                episode loop, candidate building, deciders
    answer.py                 generative and extractive answer generators
    judge.py                  answer-type-aware judge + normalizer
    energy.py                 compute, 5G transfer, payloads, radio state
    system.py                 end-to-end orchestrator
  data/                       corpus, tasks, frozen splits, training data
  docs/
    roadmap_5g_green.md       the A to H plan
    energy_model.md           every energy coefficient and its source
    eval_notes.md             per-phase measured results and decisions
    phase7_preregistration.md pre-registration for the main experiment
  models/                     trained checkpoints and bandit artifacts
  scripts/                    build, train, and evaluation scripts
  tests/                      169 tests, model-free where possible
```

## Local models only

This package must never depend on a hosted LLM API. `scripts/check_no_llm_api.py`
scans for banned imports (openai, anthropic, and similar) and known API hosts,
and `tests/test_no_llm_api.py` runs it as part of the suite. Run it directly with:

```bash
python scripts/check_no_llm_api.py
```

## Setup for running locally

I use Python 3.13 in a virtual environment. Any Python 3.10 or newer works.

```bash
cd ml_retriever
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
```

The main libraries are torch, transformers, sentence-transformers, peft,
scikit-learn, numpy, and joblib. On Apple Silicon torch uses the MPS backend
automatically. A CUDA GPU is much faster for the training steps.

## Run the tests

```bash
source .venv/bin/activate
python -m pytest -q
```

You should see 169 passing. These are mostly model-free (they use dummy
embeddings and stubbed encoders), so they run in a couple of seconds and need no
downloads.

## Run the working system on a few tasks

This uses the trained models in `models/` and prints, per task, the answer, the
gold, the judge result, and the bytes used by the bandit versus the heuristic.

```bash
source .venv/bin/activate
python scripts/run_pipeline.py --n 20 \
    --answerer models/answer-base \
    --answer_mode per_requirement
```

## Reproduce everything from scratch

If you want to rebuild the data and retrain the models end to end, run the
following in order. The training steps download the base flan-t5 weights the
first time, so they need network access. Times below are what I saw on Apple
Silicon with the MPS backend; a GPU is far faster.

```bash
source .venv/bin/activate

# 1. Build the corpus and tasks, then scale up (Phase C)
python scripts/build_seed_corpus.py
python scripts/build_seed_tasks.py
python scripts/expand_dataset_phase_c.py
python scripts/generate_synthetic_tasks.py
python scripts/curate_gold.py

# 2. Cut the frozen splits and build the per-model training data
python scripts/make_splits.py
python scripts/build_decomposer_training_data.py
python scripts/build_answer_training_data.py

# 3. Embed the corpus (needs network for the MiniLM download)
python scripts/embed_corpus.py

# 4. Train the three models
#    decomposer: long on MPS (about 15 hours for me), fast on a GPU
python scripts/train_decomposer.py --model google/flan-t5-base \
    --output_dir models/decomposer-base-lora --epochs 20 --lora \
    --lora_r 32 --lora_alpha 64 --lr 5e-4 --lora_target_modules q,v,k,o
#    answerer: about 24 minutes on MPS
python scripts/train_answer_generator.py --model google/flan-t5-base \
    --output_dir models/answer-base --epochs 12 --lora
#    bandit: a couple of minutes; lambda 0.5 is the Phase G selection
python scripts/train_bandit.py --epochs 8 --lam 0.5 --reward_mode per_step
#    external baselines (Phase E): LinUCB + Thompson, same reward and normalizer
python scripts/train_baselines.py --epochs 8 --lam 0.5
```

Note on exact reproduction: the committed `data/` directory already holds the
frozen corpus, tasks, and splits I used, so to reproduce my exact numbers you can
skip steps 1 and 2 and just embed (step 3) and train (step 4). Running steps 1
and 2 rebuilds the data deterministically and re-freezes the splits, which is
what you want for a clean-room rebuild but is not needed to match my results.

## Reproduce the individual experiments

```bash
source .venv/bin/activate

# Ground the payload model in real pages (Tier A): measures backend/pages/*.html
# (regenerate those with backend/download_pages.sh first) into
# data/measured_payloads.json. PayloadModel already defaults to the measured
# median, so this step is for verification/refresh.
python scripts/measure_real_pages.py

# Retriever recall (Phase D): compares baseline, per-requirement, entity-aware
python scripts/eval_retriever.py --k 1
python scripts/eval_retriever.py --k 5

# Main experiment on validation (Phases A, B, E, F): success, bytes, energy,
# radio, latency, and bootstrap CIs for every condition including LinUCB, LinTS,
# and the Adaptive-RAG analog. Results also written to data/phase7_results.json
python scripts/phase7_experiment.py --n 100 --split val

# Byte-penalty selection (Phase G): the lambda sweep and its cliff
python scripts/sweep_lambda.py --epochs 8 --lams 0.1 0.25 0.5 1 2 4

# Retrieval-noise characterization (Phase F): when adaptivity helps
python scripts/retrieval_noise_sweep.py --epochs 8 --noises 0 0.1 0.2 0.4 0.6

# Radio-coefficient sensitivity (Tier B): robustness across cited 5G ranges
python scripts/radio_sensitivity.py

# End-to-end on REAL live web pages (Tier A/B): measured byte and energy savings
python scripts/end_to_end_web.py

# Multi-seed variance (Phase G): stability of each learned policy across seeds
python scripts/multiseed.py --seeds 0 1 2 3 4 --epochs 8 --lam 0.5

# The single frozen test-split run (Phase G). Run this ONCE, only after every
# config is locked. It prints a warning banner because it consumes the test set.
python scripts/phase7_experiment.py --n 200 --split test

# Generate the 4 paper figures (reads the frozen-test results + measured data)
python scripts/make_figures.py     # writes figures/fig1..fig4 .png at 300 dpi
```

## The four paper figures

`scripts/make_figures.py` writes four 300-dpi figures to `figures/`, using the
frozen test-split results (`data/phase7_results_test.json`) plus the measured
payload and real-page data:

- `fig1_pareto.png`: success versus energy. The adaptive methods cluster at the
  low-energy, high-success corner while fixed budgets and full-page load spread
  out to the right. The Pareto story in one plot.
- `fig2_crossover.png`: compute versus 5G transfer energy across payload realism;
  transfer overtakes compute at real page scale.
- `fig3_radio_tail.png`: 5G RRC radio and tail energy by condition, tight-loop
  versus fast-dormancy; the radio-state contribution.
- `fig4_realpage_savings.png`: measured byte savings on real live web pages
  (about 93 to 98 percent).

## Where to read the details

- `docs/eval_notes.md` has the measured result and the decision for every phase,
  including the retriever failure analysis and the corrected Phase 7 table.
- `docs/energy_model.md` has every energy coefficient, its unit, its source, and
  the compute-versus-transfer crossover, plus the Phase F radio addendum.
- `docs/roadmap_5g_green.md` is the full A to H plan with acceptance gates.

## What is left

- The rest of Phase G: a consolidated ablation table and a reproducibility
  appendix. The multi-seed variance runs and the single final test-split run are
  now done.
- Phase H: the paper (related work, the Pareto and radio figures, limitations).
- Two known ceilings I am honest about: narrative tasks have no training coverage
  and currently fail, and two retrieval near-synonyms miss at rank 1.
- One methodological item raised by the multi-seed result: either stabilize the
  SGD bandit or adopt LinUCB as the reported learned policy (my recommendation).
```
