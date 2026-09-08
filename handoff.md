# EcoBudget Handoff Context

## Purpose

This file records implementation context as EcoBudget deliverables are completed. Future work should update this file whenever a deliverable from `teammate-1-plan.md` is achieved, changed, or handed off.

## Current Source Plan

- Plan file used: `teammate-1-plan.md`
- Active role: Teammate 1
- Ownership area: ML + Retrieval

## Deliverable Progress

### Completed: `task_decomposer` baseline

Date: 2026-09-03

Implemented a dependency-free baseline task decomposition module at `ecobudget/task_decomposer.py`.

What it provides:

- `Requirement` dataclass for atomic information needs.
- `DecomposedTask` dataclass for structured task output.
- `RuleAssistedTaskDecomposer` baseline implementation.
- `decompose_task()` convenience function.
- Validation through `validate_decomposed_task()`.
- CLI usage through `python -m ecobudget.task_decomposer "<query>"`.
- Unit tests in `tests/test_task_decomposer.py`.

Supported initial behavior:

- Detects task type for comparison, recommendation, specification lookup, multi-source research, and generic information tasks.
- Extracts entities from common comparison phrasing such as `A and B` and `A vs B`.
- Extracts attributes from phrases such as `based on`, `on`, `for`, and `considering`.
- Generates an entity-by-attribute requirement grid.
- Preserves simple constraints such as budget, location, and sustainability preference.
- Emits warnings when fallback inference is used.

Important context:

- This is intentionally a rule-assisted baseline, not the final ML decomposer.
- The final project plan still calls for a local ML-based decomposer, such as FLAN-T5 or a similarly lightweight instruction-following model.
- The current module establishes the schema and pipeline contract so retriever, evidence tracker, and policy work can proceed.
- Ground truth is not used during decomposition or retrieval-time logic.

Verification:

- `python3 -m unittest tests/test_task_decomposer.py` passes.
- `python3 -m ecobudget.task_decomposer "Compare iPhone 15 and Galaxy S24 based on price, battery capacity, and display refresh rate."` returns two entities, three attributes, and six atomic requirements.

Next likely deliverable:

- Add local ML-backed decomposition behind the same schema, or proceed to semantic representation and retrieval ranking if the team wants the pipeline skeleton first.

### Completed: Ollama-backed `task_decomposer`

Date: 2026-09-06

Replaced the active rule-assisted baseline with a local model-backed task decomposer at `ecobudget/task_decomposer.py`.

What it provides:

- `OllamaTaskDecomposer` as the default decomposer.
- `OllamaClient` using the local HTTP API at `http://localhost:11434/api/generate`.
- Default model tag: `qwen2.5:3b`.
- Same public `Requirement` and `DecomposedTask` schema used by the baseline.
- Strict JSON parsing and schema validation.
- No rule-based fallback for extracting entities or attributes from the user query.
- One model-only repair attempt when the first model response is invalid or incomplete.
- CLI usage through `python3 -m ecobudget.task_decomposer "<query>" --model qwen2.5:3b`.
- Diagnostics mode through `--diagnostics` for inspecting invalid model output without accepting it.
- Evaluation support through `EvaluationCase`, `DecompositionEvaluationReport`, and `evaluate_task_decomposer()`.

Metric reporting support:

- `valid_json_rate`: measured from model diagnostics.
- `requirement_extraction_accuracy`: measured against manually labeled expected requirements.
- `incomplete_decomposition_rate`: measured from validation failures and missing expected requirements.
- `downstream_task_success`: supported when downstream retrieval/answering provides success labels; not measurable from the decomposer alone.

Live smoke check with local Ollama:

- `what is the weather in bolivia right now?` passed validation.
- `what is percy jackson?` passed validation.
- `what is the price of thar?` passed validation.

Smoke-test metrics for the three manually checked examples:

- `valid_json_rate`: 3/3.
- `requirement_extraction_accuracy`: 3/3 against the expected single requirement for each query.
- `incomplete_decomposition_rate`: 0/3.
- `downstream_task_success`: not yet measured because downstream retrieval/answering is not implemented in this deliverable.

Verification:

- `python3 -m unittest tests/test_task_decomposer.py` passes.
- Live local-Ollama CLI checks passed for weather, general information, and price lookup query shapes.

Important context:

- The decomposer asks the model for structured JSON, validates it, and fails if the final model output remains invalid or incomplete.
- A repair pass is still model-based; it does not use handcrafted query parsing.
- Constraint normalization promotes shared requirement constraints to the top-level task and copies global constraints into each requirement for easier downstream use.
