# ml_retriever — Teammate 1 (ML + Retrieval)

EcoBudget's task-decomposition, requirement-aware-retrieval, evidence-tracking,
adaptive-policy, and answer-generation workstream. See `../plan.md` for the
full phased roadmap.

**Local models only.** This package must never depend on a hosted LLM API
(that's why `teammate-1-ml-retriever/` was scrapped). `scripts/check_no_llm_api.py`
enforces this by scanning for banned imports (`openai`, `anthropic`, etc.) and
known API hosts; it's run automatically by `tests/test_no_llm_api.py`.

## Setup

```bash
cd ml_retriever
pip install -e ".[dev]"
python -m spacy download en_core_web_sm   # needed from Phase 2 onward, not Phase 0
```

## Run tests

```bash
python -m pytest -v
```

## Layout

```
ml_retriever/
  pyproject.toml
  ml_retriever/
    __init__.py
    types.py          Requirement, Passage, AnswerResult dataclasses
    interfaces.py      EvidenceTracker, AnswerGenerator protocols
    decomposer.py       Phase 2: serialization, HeuristicDecomposer, TaskDecomposer
  data/
    README.md          Phase 1 dataset documentation (read this first)
    corpus.jsonl        84 sourced passages, 9 topics (target: 200-500)
    tasks_seed.json      36 hand-written seed tasks across those 9 topics
    tasks.json             180 tasks: seed + template-generated synthetic
    splits.json / .FROZEN  frozen train/val/test split, grouped by seed
    decomposer_train.jsonl  120 (question, target) pairs, from train split
    decomposer_val.jsonl     35 (question, target) pairs, from val split
    decomposer_eval_results.jsonl  appended to by eval_decomposer.py runs
  scripts/
    check_no_llm_api.py
    build_seed_corpus.py
    build_seed_tasks.py
    generate_synthetic_tasks.py
    make_splits.py
    validate_corpus_task_coverage.py
    embed_corpus.py     writes embeddings into corpus.jsonl (needs real network)
    build_decomposer_training_data.py  writes decomposer_{train,val}.jsonl
    train_decomposer.py   fine-tunes flan-t5-{small,base}, optional LoRA (needs real network)
    eval_decomposer.py     exact-match/F1/latency eval, heuristic or trained
  tests/
    test_types.py
    test_interfaces.py
    test_no_llm_api.py    now covers both ml_retriever/ and scripts/
    test_data.py           Phase 1 data validation (17 tests)
    test_decomposer.py      Phase 2: serialization, parsing, heuristic baseline (16 tests)
```

## Status

Phase 0 complete: dataclasses, interfaces, no-LLM-API guard.

Phase 1 mostly complete: 84/200-500 sourced passages across 9 topics
(phones, laptops, electronics, travel, buildings, geography, countries,
running shoes, vehicles — spanning all five categories in
project_overview, not just gadget comparisons), 180/150-200 tasks (36
seed + 144 template-synthetic), frozen leakage-safe split, full
requirement→passage coverage, and a regression test guarding topic
diversity. Embeddings not yet computed (needs a non-sandboxed machine —
see `data/README.md`).

Phase 2 scaffolded, not yet trained:
- `ml_retriever/decomposer.py` defines the serialization format
  (`entity|attribute ## entity|attribute`), a robust parser (drops
  malformed segments rather than raising), a `HeuristicDecomposer`
  baseline (keyword + gazetteer matching, no model), and `TaskDecomposer`
  (wraps a fine-tuned flan-t5 model, optional LoRA adapter).
- The heuristic baseline is **measured, not assumed**: 77.1% exact-match
  accuracy / 90.5% F1 on the val split (`decomposer_eval_results.jsonl`).
  That's the real number Phase 2's fine-tuned model needs to beat to
  hit the plan's >90% exact-match target — it is not a strawman.
- `scripts/build_decomposer_training_data.py`, `train_decomposer.py`,
  and `eval_decomposer.py` are written and wired to the frozen train/val
  split (test split is not selectable from eval_decomposer.py — see its
  docstring). `train_decomposer.py` needs network access to download
  base flan-t5 weights, so **it hasn't been run in this sandbox**; run
  it on a normal machine after `pip install -e ".[train]"`.
- Not yet done: actually training flan-t5-small and flan-t5-base,
  running the two-model-size comparison, and confirming the >90%
  exact-match target on val.

57/57 tests passing (41 from Phase 0/1 + 16 new for the decomposer:
serialization round-trips, malformed-output handling, heuristic behavior
on both synthetic and real val-split questions).

## Next steps (to run on a machine with network access)

```bash
pip install -e ".[train]"          # adds peft, datasets
python -m spacy download en_core_web_sm
python scripts/embed_corpus.py     # Phase 1 leftover: embed the corpus

python scripts/train_decomposer.py --model google/flan-t5-small \
    --output_dir models/decomposer-small --epochs 5
python scripts/train_decomposer.py --model google/flan-t5-base \
    --output_dir models/decomposer-base-lora --epochs 5 --lora

python scripts/eval_decomposer.py --heuristic --tag heuristic
python scripts/eval_decomposer.py --model_path models/decomposer-small --tag small
python scripts/eval_decomposer.py --model_path google/flan-t5-base \
    --adapter_path models/decomposer-base-lora --tag base+lora
# compare all three rows in data/decomposer_eval_results.jsonl
```
