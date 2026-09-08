# EcoBudget Task Decomposer

This deliverable converts a natural-language search or research request into a validated JSON task that downstream retrieval and answering components can use.

It runs locally through Ollama and defaults to `qwen2.5:3b`. The decomposer does not use a rule-based extraction fallback: invalid or incomplete model output receives one model-based repair attempt, then fails clearly if it is still unusable.

## What It Produces

For each query, the decomposer returns:

- `task_type`: the kind of request, such as `comparison` or `information_task`.
- `entities`: the things being researched.
- `attributes`: the facts needed about those things.
- `requirements`: one atomic requirement for every entity and attribute pair.
- `constraints`: shared details such as freshness, budget, location, or preferences.
- `answer_format`: the expected response shape.

For example, `what is the weather in bolivia right now?` should identify `Bolivia` as the entity, `current weather` as the attribute, and `right now` as a freshness constraint.

## Prerequisites

- Python 3
- Ollama installed locally
- The Qwen model available locally:

```bash
ollama pull qwen2.5:3b
```

## Run The Automated Tests

From the project directory, run:

```bash
python3 -m unittest tests/test_task_decomposer.py -v
```

These tests use a mocked model client, so Ollama does not need to be running. They check valid decomposition, schema completeness, invalid JSON failure, incomplete-output failure, and the model-based repair attempt.

## Run A Live Model Test

Start Ollama in one terminal if it is not already running:

```bash
ollama serve
```

Then, in the project directory, run a query:

```bash
python3 -m ecobudget.task_decomposer \
  "what is the weather in bolivia right now?" \
  --model qwen2.5:3b \
  --diagnostics
```

Try the other supported request shapes too:

```bash
python3 -m ecobudget.task_decomposer "what is percy jackson?"
python3 -m ecobudget.task_decomposer "what is the price of thar?"
python3 -m ecobudget.task_decomposer "Compare iPhone 15 and Galaxy S24 based on price and battery capacity."
```

A successful run prints JSON with non-empty `entities`, `attributes`, `requirements`, and `answer_format`. For comparisons, the requirement count must equal the number of entities multiplied by the number of attributes.

## Failure Rules

The decomposer rejects output when it is not JSON, does not match the schema, has empty entities or attributes, lacks required entity-attribute pairs, or otherwise fails validation. It will make one model-only repair attempt before returning an error.

Test this behavior with an empty request:

```bash
python3 -m ecobudget.task_decomposer ""
```

This should fail before the model is called.

## Quality Metrics

`evaluate_task_decomposer()` supports these measures against manually labeled evaluation cases:

- Valid JSON rate
- Requirement extraction accuracy
- Incomplete decomposition rate
- Downstream task success, once retrieval and answering exist

The current smoke check covered weather, general information, and price queries: valid JSON was 3/3, requirement extraction was 3/3 for the expected single requirement in each case, and incomplete decomposition was 0/3. Downstream task success is not measurable until later pipeline stages are implemented.

## Project Files

- `ecobudget/task_decomposer.py`: Ollama client, prompts, validation, CLI, and evaluation helpers.
- `tests/test_task_decomposer.py`: mocked-model tests.
- `handoff.md`: teammate context and completed-deliverable record.
- `teammate-1-plan.md`: Teammate 1 work plan.
