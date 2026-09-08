# EcoBudget Teammate 1 Plan: ML + Retrieval

## Role Summary

Teammate 1 owns the machine-learning and retrieval intelligence for EcoBudget. The goal is to build the part of the system that understands a user's research task, identifies the information needed to complete it, retrieves only useful evidence, tracks what is still missing, and learns when additional retrieval is worth the data cost.

EcoBudget should be treated as a research prototype, not a generic chatbot. The main research goal is to test whether adaptive retrieval can reduce web data transfer while maintaining acceptable task completion quality.

## Project Understanding

### Core Problem

Modern AI and web assistants often retrieve far more web data than necessary. For realistic information-heavy tasks, such as product comparisons, travel research, specification lookup, and multi-source decision-making, the system may download full webpages even though only a small subset of information is needed.

EcoBudget asks:

> Can an AI-powered web research system retrieve only the information necessary to complete a user's task, reducing data transfer while maintaining acceptable answer quality?

### Current Direction

The redesigned EcoBudget system should become an ML-driven, data-aware web research assistant. It should:

- Understand a user's information task.
- Decompose the task into explicit information requirements.
- Search and rank candidate sources.
- Retrieve selected evidence rather than entire pages whenever possible.
- Track which requirements are supported by evidence.
- Decide adaptively whether to retrieve more information or stop.
- Compare its performance against normal retrieval, fixed-budget retrieval, and heuristic adaptive retrieval.

### Teammate 1 Ownership

Teammate 1 is responsible for:

- `task_decomposer`
- `retriever`
- `evidence_tracker`
- `bandit_policy`
- ML evaluation

Teammate 2 owns the web pipeline, fetching, parsing, byte measurement, carbon estimation, baselines, experiment runner, and frontend. Teammate 1 must therefore expose clean interfaces so Teammate 2 can plug in real web data, byte counts, caching, and experiment orchestration.

## Teammate 1 Objectives

1. Build an ML-based task understanding module that converts user tasks into structured information requirements.
2. Build retrieval ranking logic that prioritizes sources/passages likely to satisfy missing requirements.
3. Build an evidence tracker that records which requirements have supporting evidence and which remain unanswered.
4. Build an adaptive retrieval policy that decides whether to retrieve more information or stop.
5. Implement a contextual bandit policy that learns retrieval decisions from task outcomes and data cost.
6. Evaluate whether the learned policy improves the quality-data trade-off compared with fixed budgets and heuristic retrieval.

## System Components

## 1. Task Decomposer

### Purpose

Convert a natural-language user task into a structured list of entities, attributes, constraints, and output expectations.

Example input:

```text
Compare iPhone 15 and Galaxy S24 based on price, battery capacity, and display refresh rate.
```

Expected structured output:

```json
{
  "task_type": "comparison",
  "entities": ["iPhone 15", "Galaxy S24"],
  "attributes": ["price", "battery capacity", "display refresh rate"],
  "requirements": [
    {"entity": "iPhone 15", "attribute": "price"},
    {"entity": "iPhone 15", "attribute": "battery capacity"},
    {"entity": "iPhone 15", "attribute": "display refresh rate"},
    {"entity": "Galaxy S24", "attribute": "price"},
    {"entity": "Galaxy S24", "attribute": "battery capacity"},
    {"entity": "Galaxy S24", "attribute": "display refresh rate"}
  ],
  "answer_format": "comparison"
}
```

### Initial Approach

Use a local instruction-following model, such as FLAN-T5 or a similarly lightweight local model, for task decomposition. Avoid building the final system around handcrafted regex rules because the project history already showed that regex-based matching is too brittle for realistic tasks.

### Responsibilities

- Identify the task type:
  - comparison
  - specification lookup
  - recommendation
  - multi-source research
  - travel/product decision task
- Extract entities.
- Extract requested attributes.
- Generate atomic requirements.
- Preserve user constraints, such as price limits, location, dates, sustainability preferences, or intended use case.
- Return a structured JSON-like representation that other modules can consume.

### Implementation Milestones

1. Define the output schema for decomposed tasks.
2. Build a baseline rule-assisted decomposer for early testing only.
3. Add local ML-based decomposition.
4. Add validation to ensure outputs include entities, attributes, and requirements.
5. Create a small manually labeled decomposition test set.
6. Measure decomposition quality before using it in final retrieval experiments.

### Evaluation

Measure:

- Entity extraction accuracy.
- Attribute extraction accuracy.
- Requirement completeness.
- Rate of invalid structured outputs.
- Error types, such as missing entity, merged attributes, hallucinated requirement, or wrong task type.

## 2. Semantic Representation

### Purpose

Represent tasks, requirements, sources, and passages in a way that supports semantic matching.

### Responsibilities

- Create text representations for each requirement.
- Compute embeddings for requirements and candidate passages.
- Compare missing requirements against source snippets/passages.
- Help rank retrieval candidates by likely usefulness.

### Possible Representations

For a requirement:

```text
iPhone 15 price
Galaxy S24 battery capacity
MacBook Air M3 weight
```

For a passage:

```text
The Galaxy S24 has a 4,000 mAh battery and supports a 120Hz adaptive display.
```

### Recommended Approach

Start with a lightweight sentence-transformer style embedding model or another local semantic similarity model. If local model setup is too heavy, begin with TF-IDF/BM25 as a fallback baseline, but keep the interface compatible with embeddings.

### Deliverables

- Requirement embedding function.
- Passage/source embedding function.
- Similarity scoring function.
- Ranked candidate list for each missing requirement.

## 3. Retriever

### Purpose

Choose which candidate sources or passages should be retrieved next.

The retriever should not own low-level network fetching. Teammate 2 will likely provide candidate source metadata, snippets, parsed text chunks, byte counts, and cached content. Teammate 1 owns ranking and selection logic.

### Inputs

- Decomposed task.
- Current evidence state.
- Candidate source list.
- Candidate snippets or passages.
- Data cost estimates, if available.
- Current retrieval budget, if applicable.

### Outputs

- Ranked candidate retrieval actions.
- Expected relevance/usefulness score.
- Estimated requirement coverage.
- Estimated byte cost.

### Retrieval Action Types

Initial action set:

```text
retrieve_source_1
retrieve_source_2
retrieve_source_3
retrieve_next_passage
retrieve_more_from_same_source
stop
```

Later action set:

```text
retrieve_best_for_missing_requirement
retrieve_cheapest_relevant_passage
retrieve_high_confidence_source
retrieve_diverse_source
retrieve_more_context
stop
```

### Ranking Criteria

Rank candidates based on:

- Semantic relevance to missing requirements.
- Number of missing requirements likely covered.
- Source reliability signal, if available.
- Redundancy with already retrieved evidence.
- Estimated bytes required.
- Expected value per byte.

### Initial Formula

Use a transparent heuristic score before training the bandit:

```text
candidate_score =
  relevance_score
  + coverage_score
  + reliability_score
  - redundancy_penalty
  - byte_cost_penalty
```

This creates a non-learning adaptive baseline and supplies useful features for the contextual bandit.

## 4. Evidence Tracker

### Purpose

Maintain a live state of what the system knows and what is still missing.

Example:

```text
Requirement                         Status       Evidence
iPhone 15 / price                   found        source A
iPhone 15 / battery capacity        missing      -
iPhone 15 / display refresh rate    found        source B
Galaxy S24 / price                  found        source C
Galaxy S24 / battery capacity       found        source C
Galaxy S24 / display refresh rate   missing      -
```

### Responsibilities

- Store all atomic requirements.
- Attach retrieved evidence to requirements.
- Score evidence strength.
- Detect missing requirements.
- Detect conflicting evidence.
- Track whether enough evidence exists to answer.
- Provide state features to the adaptive policy.

### Evidence Fields

Each evidence item should include:

```json
{
  "requirement_id": "galaxy_s24_battery_capacity",
  "source_id": "source_003",
  "source_url": "https://example.com",
  "text": "Galaxy S24 includes a 4,000 mAh battery.",
  "extracted_value": "4,000 mAh",
  "confidence": 0.82,
  "bytes_used": 1840,
  "retrieval_step": 3
}
```

### Evidence Scoring Signals

Use multiple signals rather than QA confidence alone:

- Requirement-passage semantic similarity.
- Presence of entity and attribute.
- Extracted answer/value form.
- Agreement across sources.
- Source reliability.
- Specificity of evidence.
- Whether the passage directly answers the requirement.

### Stop Readiness

The evidence tracker should expose:

```text
coverage_ratio
missing_requirement_count
average_evidence_confidence
conflict_count
bytes_used
retrieval_steps
ready_to_answer
```

The bandit should use these signals, but the evidence tracker should not use ground truth answers during retrieval.

## 5. Adaptive Heuristic Policy

### Purpose

Create a strong non-learning adaptive baseline before adding the contextual bandit.

This baseline is important because the final claim should not merely be "we used a bandit." The experiment must show whether learning improves the retrieval decision compared with a reasonable heuristic.

### Policy Logic

Continue retrieving when:

- Required information is still missing.
- Current evidence confidence is low.
- Evidence conflicts exist.
- A high-value candidate remains available.
- The byte budget has not been exceeded.

Stop when:

- All required information has sufficient evidence.
- Additional candidates have low expected value.
- The marginal value per byte is too low.
- A maximum budget or retrieval-step limit is reached.

### Deliverables

- `AdaptiveHeuristicPolicy`
- Configurable thresholds.
- Logs explaining each stop/retrieve decision.
- Metrics for bytes used and requirement coverage at stop time.

## 6. Contextual Bandit Policy

### Purpose

Learn when retrieving more information is worth the additional data cost.

The bandit does not answer the question. It controls retrieval decisions.

### Context Features

Candidate features:

- Candidate relevance to missing requirements.
- Estimated number of requirements covered.
- Candidate source rank.
- Candidate estimated byte cost.
- Candidate source reliability.
- Candidate redundancy score.

Task state features:

- Number of total requirements.
- Number of missing requirements.
- Coverage ratio.
- Average evidence confidence.
- Lowest evidence confidence.
- Conflict count.
- Bytes used so far.
- Retrieval steps so far.
- Remaining budget.
- Task type.
- Number of entities.
- Number of attributes.

Action history features:

- Previous retrieval actions.
- Marginal evidence gained by last action.
- Bytes spent by last action.
- Whether last action reduced missing requirements.

### Actions

Minimum viable action set:

```text
retrieve_top_candidate
retrieve_cheapest_relevant_candidate
retrieve_most_diverse_candidate
retrieve_more_context_from_current_source
stop
```

If time is limited, simplify to:

```text
retrieve_next
stop
```

Then expand after the simpler version works.

### Reward Design

Reward should encourage task success and penalize unnecessary data transfer.

Suggested reward:

```text
reward = task_success_reward - data_cost_penalty - failed_task_penalty
```

Example:

```text
if task_success:
    reward = 1.0 - alpha * normalized_bytes
else:
    reward = -1.0 - beta * normalized_bytes
```

Alternative step-level shaping:

```text
step_reward =
  evidence_gain
  - byte_penalty
  - redundancy_penalty
```

Final reward:

```text
final_reward =
  task_completion_score
  - total_byte_penalty
```

### Important Constraint

The policy must not use ground truth during retrieval. Ground truth can only be used after a run finishes to evaluate success and assign training rewards.

### Candidate Algorithms

Start simple:

- Epsilon-greedy linear bandit.
- LinUCB.
- Thompson sampling with linear reward model.

Recommended first implementation:

```text
LinUCB or epsilon-greedy logistic/linear model
```

The model should be explainable enough for a conference poster and report.

### Training Setup

1. Run baseline systems over benchmark tasks.
2. Log contexts, actions, bytes, evidence gain, and final task success.
3. Train the bandit policy from logged interactions or through repeated simulated/offline runs.
4. Evaluate on a held-out task split.
5. Compare against fixed budgets and heuristic adaptive retrieval.

### Bandit Deliverables

- Context feature builder.
- Action definitions.
- Bandit policy class.
- Training loop.
- Policy checkpoint saving/loading.
- Decision logs.
- Evaluation script or function.

## 7. Answer Generation Interface

### Purpose

Although Teammate 1 is not primarily responsible for the frontend, the ML/retrieval side must provide clean evidence for answer generation.

### Requirements

The answer generator should only use retrieved evidence. It should not invent unsupported facts.

Teammate 1 should provide:

- Structured evidence table.
- Source references.
- Missing/low-confidence requirement flags.
- Conflicts, if any.
- Final retrieval summary.

Example output to downstream answer module:

```json
{
  "task": "Compare iPhone 15 and Galaxy S24 on price, battery, and refresh rate.",
  "requirements": [...],
  "evidence": [...],
  "missing_requirements": [],
  "conflicts": [],
  "retrieval_summary": {
    "bytes_used": 18420,
    "retrieval_steps": 5,
    "sources_used": 3,
    "policy": "contextual_bandit"
  }
}
```

## 8. ML Evaluation Plan

### Evaluation Questions

Teammate 1 should answer:

1. Does ML task decomposition produce complete and valid information requirements?
2. Does retrieval ranking find useful evidence for missing requirements?
3. Does evidence tracking correctly identify whether requirements are satisfied?
4. Does adaptive retrieval stop at reasonable points?
5. Does the contextual bandit improve the task-success vs data-transfer trade-off?

### Metrics

Task decomposition:

- Entity extraction accuracy.
- Attribute extraction accuracy.
- Requirement F1.
- Invalid output rate.

Retrieval:

- Precision@k for relevant passages.
- Requirement coverage after k retrievals.
- Evidence gain per KB.
- Redundancy rate.

Evidence tracking:

- Requirement satisfaction accuracy.
- Missing requirement detection.
- Conflict detection, if applicable.
- Calibration of confidence scores.

Policy:

- Task completion rate.
- Bytes transferred.
- Retrieval actions per task.
- Success per KB.
- Regret compared with best observed policy.
- Stop-decision accuracy.

Main research comparison:

```text
task completion rate vs bytes transferred
```

## 9. Benchmark Contribution

Although benchmark design is shared, Teammate 1 should help ensure the benchmark is suitable for ML retrieval evaluation.

### Benchmark Requirements

The final benchmark should include 30-50 realistic information tasks across:

- Product comparison.
- Technical specification lookup.
- Travel or venue comparison.
- Multi-source research.
- Decision-making tasks.
- Sustainability-related comparisons.

### Teammate 1 Benchmark Responsibilities

- Define the required information fields for each task.
- Help create ground truth requirement tables.
- Mark acceptable answer values.
- Identify tasks that are ambiguous or too trivial.
- Freeze the benchmark before final experiments.

### Example Ground Truth Format

```json
{
  "task_id": "phone_compare_001",
  "task": "Compare iPhone 15 and Galaxy S24 on price, battery capacity, and refresh rate.",
  "requirements": [
    {
      "entity": "iPhone 15",
      "attribute": "price",
      "acceptable_values": ["$799", "799 USD", "starts at $799"]
    }
  ]
}
```

## 10. Integration Contract With Teammate 2

### Teammate 1 Provides

- Task decomposition schema.
- Requirement list.
- Retrieval ranking scores.
- Evidence tracker state.
- Policy action recommendation.
- Final evidence package.
- ML decision logs.

### Teammate 2 Provides

- Web search candidates.
- Fetching and parsing.
- Raw and cleaned page content.
- Byte counts.
- Data budget enforcement.
- Caching.
- Carbon estimate pipeline.
- Experiment runner.
- Frontend display.

### Required Shared Data Structures

Task object:

```json
{
  "task_id": "string",
  "user_query": "string",
  "task_type": "comparison",
  "entities": ["string"],
  "attributes": ["string"],
  "requirements": []
}
```

Candidate object:

```json
{
  "candidate_id": "string",
  "url": "string",
  "title": "string",
  "snippet": "string",
  "estimated_bytes": 0,
  "source_rank": 1
}
```

Policy decision object:

```json
{
  "action": "retrieve_top_candidate",
  "candidate_id": "string",
  "reason": "Highest expected coverage for missing battery and price requirements.",
  "context_features": {},
  "expected_value": 0.0
}
```

Evidence state object:

```json
{
  "coverage_ratio": 0.0,
  "missing_requirements": [],
  "evidence_items": [],
  "bytes_used": 0,
  "retrieval_steps": 0,
  "ready_to_answer": false
}
```

## 11. Development Phases

### Phase 1: Define Schemas and Interfaces

Tasks:

- Define task decomposition output schema.
- Define requirement object schema.
- Define candidate source/passage schema.
- Define evidence state schema.
- Define policy decision schema.

Deliverables:

- Schema documentation.
- Minimal placeholder classes or interface definitions.
- Example JSON files for 3-5 tasks.

Completion criteria:

- Teammate 2 can build around the same data structures.
- A decomposed task can flow into retrieval ranking and evidence tracking.

### Phase 2: Build Task Decomposition

Tasks:

- Implement local ML-based task decomposition.
- Build validation for model outputs.
- Add fallback baseline for simple comparison tasks.
- Create a small labeled set for decomposition evaluation.

Deliverables:

- `task_decomposer`
- Decomposition evaluation report.
- Error analysis table.

Completion criteria:

- Realistic comparison tasks produce usable requirement tables.
- Invalid or incomplete outputs are detected and logged.

### Phase 3: Build Retrieval Ranking

Tasks:

- Represent requirements and candidate passages semantically.
- Score candidates against missing requirements.
- Add redundancy detection.
- Add value-per-byte style scoring.

Deliverables:

- `retriever`
- Candidate ranking output.
- Retrieval scoring explanation logs.

Completion criteria:

- Given candidate snippets/passages, the retriever ranks evidence-rich candidates above irrelevant candidates.
- Ranking can operate using missing requirements from the evidence tracker.

### Phase 4: Build Evidence Tracker

Tasks:

- Track requirement status.
- Attach evidence to requirements.
- Compute coverage and confidence.
- Detect missing and weakly supported requirements.
- Surface readiness-to-answer state.

Deliverables:

- `evidence_tracker`
- Evidence table output.
- Requirement coverage metrics.

Completion criteria:

- The tracker correctly updates after each retrieval step.
- The tracker can identify whether the system has enough evidence to answer.

### Phase 5: Build Adaptive Heuristic Baseline

Tasks:

- Implement retrieve/stop decision rules.
- Use coverage, confidence, expected value, and byte cost.
- Add configurable thresholds.
- Log reasons for stopping or continuing.

Deliverables:

- `adaptive_heuristic_policy`
- Decision trace logs.
- Baseline results on early benchmark tasks.

Completion criteria:

- The heuristic baseline can complete a full retrieval loop.
- It provides a meaningful comparison point for the bandit.

### Phase 6: Build Contextual Bandit

Tasks:

- Define context feature vector.
- Define action space.
- Implement first bandit algorithm.
- Train from logged interactions or repeated benchmark runs.
- Save and load learned policy state.

Deliverables:

- `bandit_policy`
- Feature builder.
- Training loop.
- Policy checkpoint.
- Decision trace logs.

Completion criteria:

- The policy chooses retrieval actions based on state features.
- The policy includes a real STOP action.
- The policy is evaluated against fixed budgets and heuristic retrieval.

### Phase 7: ML Evaluation and Error Analysis

Tasks:

- Run decomposition evaluation.
- Run retrieval ranking evaluation.
- Run evidence tracking checks.
- Compare policy conditions.
- Document failure cases.

Deliverables:

- ML evaluation results.
- Error analysis.
- Plots/tables for task completion vs bytes transferred.
- Notes for final report/poster.

Completion criteria:

- Results show whether the learned policy improves the quality-data trade-off.
- Failure cases are documented honestly.

## 12. Experimental Conditions Relevant to Teammate 1

Teammate 1 should support the following conditions:

```text
Normal retrieval
Fixed-10KB
Fixed-25KB
Fixed-50KB
Fixed-100KB
Adaptive heuristic
Contextual bandit
```

Teammate 2 may implement the runner and byte enforcement, but Teammate 1 must ensure retrieval and policy modules can operate under each condition.

## 13. Risks and Mitigations

### Risk: The decomposer produces incomplete requirements

Mitigation:

- Add schema validation.
- Use examples in prompts.
- Keep a fallback parser for simple comparison tasks.
- Log decomposition confidence and failure cases.

### Risk: Retrieval ranking rewards relevance instead of usefulness

Mitigation:

- Rank against missing requirements, not just the full query.
- Track evidence gain after each retrieval.
- Penalize redundancy.
- Evaluate requirement coverage, not only passage similarity.

### Risk: QA confidence is mistaken for correctness

Mitigation:

- Use multiple evidence signals.
- Require entity, attribute, and answer-form checks.
- Support agreement across sources.
- Keep final correctness evaluation separate from retrieval-time confidence.

### Risk: The bandit becomes superficial

Mitigation:

- Give the bandit real actions, including STOP.
- Log every context-action-reward tuple.
- Compare against fixed budgets and adaptive heuristic baselines.
- Show whether learning improves the data-quality trade-off.

### Risk: The benchmark is too small or too ambiguous

Mitigation:

- Help design realistic 30-50 task benchmark.
- Freeze tasks before final experiments.
- Remove ambiguous tasks before evaluation.
- Keep ground truth separate from retrieval-time logic.

## 14. Suggested Timeline

### Week 1

- Finalize schemas with Teammate 2.
- Build task decomposition prototype.
- Create 5-10 labeled decomposition examples.

### Week 2

- Implement semantic requirement and passage scoring.
- Build initial retriever ranking.
- Create evidence tracker prototype.

### Week 3

- Connect decomposer, retriever, and evidence tracker into a retrieval loop.
- Implement adaptive heuristic policy.
- Test on 5-10 realistic tasks.

### Week 4

- Define contextual bandit features and actions.
- Implement first bandit algorithm.
- Start logging training data.

### Week 5

- Train and evaluate bandit policy.
- Compare against fixed budgets and heuristic retrieval.
- Run error analysis.

### Week 6

- Freeze benchmark.
- Prepare final ML evaluation results.
- Produce tables/plots for poster and report.
- Document limitations and failure cases.

## 15. Final Teammate 1 Deliverables

By the end of the project, Teammate 1 should deliver:

- ML task decomposition module.
- Structured requirement schema.
- Semantic representation and ranking module.
- Selective retrieval decision logic.
- Evidence tracker.
- Adaptive heuristic retrieval policy.
- Contextual bandit retrieval policy.
- Training/evaluation logs.
- ML evaluation results.
- Error analysis.
- Contribution to benchmark ground truth.
- Clear interface documentation for Teammate 2.

## 16. Definition of Done

Teammate 1's work is complete when:

- A realistic user task can be decomposed into entities, attributes, and requirements.
- Candidate evidence can be ranked against missing requirements.
- Retrieved evidence can be attached to specific requirements.
- The system can decide whether to retrieve more information or stop.
- The contextual bandit makes real retrieval decisions and includes a STOP action.
- The learned policy is compared against fixed-budget and heuristic baselines.
- Evaluation reports task completion, bytes transferred, retrieval actions, and failure cases.
- No ground truth is used during retrieval decisions.
- The outputs are clean enough for Teammate 2 to integrate into the full EcoBudget prototype.

## 17. Research Story for Teammate 1

Teammate 1's contribution should support this claim:

> EcoBudget does not simply retrieve less data by using a smaller fixed budget. It learns task-aware retrieval behavior by tracking missing information and deciding whether additional evidence is worth its data cost.

The strongest result would be:

> The contextual bandit achieves similar task completion to normal retrieval while transferring less data, and performs better than simple fixed-budget retrieval at the same or lower data cost.

This claim should only be used if the experimental results support it.
