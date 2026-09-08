"""Model-backed task decomposition for EcoBudget.

This module converts a natural-language research task into explicit atomic
information requirements. The default implementation uses a local Ollama model
through the HTTP API and fails when the model output is invalid or incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
ALLOWED_TASK_TYPES = {
    "comparison",
    "specification_lookup",
    "recommendation",
    "multi_source_research",
    "travel_product_decision",
    "information_task",
}


class TaskDecompositionError(ValueError):
    """Raised when a query cannot be decomposed into usable requirements."""


class ModelClient(Protocol):
    """Small interface used by the decomposer and tests."""

    def generate(self, prompt: str) -> str:
        """Return model text for a prompt."""


@dataclass(frozen=True)
class Requirement:
    """One atomic information need for a task."""

    id: str
    entity: str
    attribute: str
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecomposedTask:
    """Structured task representation consumed by retrieval modules."""

    user_query: str
    task_type: str
    entities: list[str]
    attributes: list[str]
    requirements: list[Requirement]
    answer_format: str
    constraints: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requirements"] = [requirement.to_dict() for requirement in self.requirements]
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class DecompositionDiagnostics:
    """Result of one decomposition attempt, including validation state."""

    user_query: str
    raw_response: str
    valid_json: bool
    validation_errors: list[str] = field(default_factory=list)
    task: DecomposedTask | None = None

    @property
    def is_complete(self) -> bool:
        return self.valid_json and self.task is not None and not self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "raw_response": self.raw_response,
            "valid_json": self.valid_json,
            "validation_errors": self.validation_errors,
            "is_complete": self.is_complete,
            "task": self.task.to_dict() if self.task else None,
        }


@dataclass(frozen=True)
class EvaluationCase:
    """Manually labeled case for measuring decomposer quality."""

    user_query: str
    expected_entities: list[str] = field(default_factory=list)
    expected_attributes: list[str] = field(default_factory=list)
    expected_requirements: list[tuple[str, str]] = field(default_factory=list)
    downstream_success: bool | None = None


@dataclass(frozen=True)
class DecompositionEvaluationReport:
    """Aggregate quality metrics for a decomposer run."""

    total_cases: int
    valid_json_rate: float
    requirement_extraction_accuracy: float | None
    incomplete_decomposition_rate: float
    downstream_task_success: float | None
    diagnostics: list[DecompositionDiagnostics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diagnostics"] = [
            {
                "user_query": diagnostic.user_query,
                "valid_json": diagnostic.valid_json,
                "validation_errors": diagnostic.validation_errors,
                "is_complete": diagnostic.is_complete,
            }
            for diagnostic in self.diagnostics
        ]
        return payload


class OllamaClient:
    """Minimal client for Ollama's local HTTP generation API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise TaskDecompositionError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TaskDecompositionError("Ollama returned a non-JSON API response.") from exc

        model_text = data.get("response")
        if not isinstance(model_text, str) or not model_text.strip():
            raise TaskDecompositionError("Ollama returned an empty model response.")
        return model_text


class OllamaTaskDecomposer:
    """Task decomposer that uses a local Ollama instruction model."""

    def __init__(
        self,
        *,
        client: ModelClient | None = None,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        max_attempts: int = 2,
    ) -> None:
        self.client = client or OllamaClient(model=model, host=host)
        self.max_attempts = max(1, max_attempts)

    def decompose(self, user_query: str) -> DecomposedTask:
        diagnostics = self.decompose_with_diagnostics(user_query)
        if not diagnostics.valid_json:
            raise TaskDecompositionError("Model output was not valid JSON.")
        if diagnostics.validation_errors:
            raise TaskDecompositionError("; ".join(diagnostics.validation_errors))
        if diagnostics.task is None:
            raise TaskDecompositionError("Model output could not be converted to a task.")
        return diagnostics.task

    def decompose_with_diagnostics(self, user_query: str) -> DecompositionDiagnostics:
        query = _clean_space(user_query)
        if not query:
            raise TaskDecompositionError("User query cannot be empty.")

        prompt = _build_decomposition_prompt(query)
        last_diagnostic: DecompositionDiagnostics | None = None
        for attempt_index in range(self.max_attempts):
            raw_response = self.client.generate(prompt)
            diagnostic = _diagnose_model_response(query, raw_response)
            if diagnostic.is_complete:
                return diagnostic

            last_diagnostic = diagnostic
            if attempt_index < self.max_attempts - 1:
                prompt = _build_repair_prompt(
                    query,
                    raw_response=raw_response,
                    validation_errors=diagnostic.validation_errors,
                )

        if last_diagnostic is None:
            raise TaskDecompositionError("Model was not called.")
        return last_diagnostic


def _diagnose_model_response(user_query: str, raw_response: str) -> DecompositionDiagnostics:
    try:
        payload = _parse_json_object(raw_response)
    except json.JSONDecodeError:
        return DecompositionDiagnostics(
            user_query=user_query,
            raw_response=raw_response,
            valid_json=False,
            validation_errors=["model output is not valid JSON"],
        )

    try:
        task = _task_from_payload(payload, expected_query=user_query)
    except TaskDecompositionError as exc:
        return DecompositionDiagnostics(
            user_query=user_query,
            raw_response=raw_response,
            valid_json=True,
            validation_errors=[str(exc)],
        )

    validation_errors = validate_decomposed_task(task)
    return DecompositionDiagnostics(
        user_query=user_query,
        raw_response=raw_response,
        valid_json=True,
        validation_errors=validation_errors,
        task=task if not validation_errors else None,
    )


def validate_decomposed_task(task: DecomposedTask) -> list[str]:
    """Return validation errors for a decomposed task."""

    errors: list[str] = []
    if not task.user_query:
        errors.append("missing user_query")
    if not task.task_type:
        errors.append("missing task_type")
    elif task.task_type not in ALLOWED_TASK_TYPES:
        errors.append(f"unknown task_type: {task.task_type}")
    if not task.entities:
        errors.append("no entities extracted")
    if not task.attributes:
        errors.append("no attributes extracted")
    if not task.requirements:
        errors.append("no requirements generated")
    if not task.answer_format:
        errors.append("missing answer_format")

    expected_count = len(task.entities) * len(task.attributes)
    if task.entities and task.attributes and len(task.requirements) != expected_count:
        errors.append(
            f"requirement count mismatch: expected {expected_count}, got {len(task.requirements)}"
        )

    valid_pairs = {
        (_normalize_for_match(entity), _normalize_for_match(attribute))
        for entity in task.entities
        for attribute in task.attributes
    }
    requirement_pairs: set[tuple[str, str]] = set()
    requirement_ids: set[str] = set()
    for requirement in task.requirements:
        if not requirement.id:
            errors.append("requirement missing id")
        elif requirement.id in requirement_ids:
            errors.append(f"duplicate requirement id: {requirement.id}")
        requirement_ids.add(requirement.id)

        if not requirement.entity:
            errors.append(f"{requirement.id or '<missing id>'} missing entity")
        if not requirement.attribute:
            errors.append(f"{requirement.id or '<missing id>'} missing attribute")

        pair = (
            _normalize_for_match(requirement.entity),
            _normalize_for_match(requirement.attribute),
        )
        if pair in requirement_pairs:
            errors.append(
                f"duplicate requirement pair: {requirement.entity} / {requirement.attribute}"
            )
        requirement_pairs.add(pair)
        if valid_pairs and pair not in valid_pairs:
            errors.append(
                f"requirement pair not present in entity/attribute grid: "
                f"{requirement.entity} / {requirement.attribute}"
            )

    return errors


def decompose_task(user_query: str) -> DecomposedTask:
    """Convenience function for the default Ollama-backed decomposer."""

    return OllamaTaskDecomposer().decompose(user_query)


def evaluate_task_decomposer(
    decomposer: OllamaTaskDecomposer,
    cases: list[EvaluationCase],
) -> DecompositionEvaluationReport:
    """Evaluate decomposition quality on manually labeled cases."""

    if not cases:
        raise ValueError("At least one evaluation case is required.")

    diagnostics: list[DecompositionDiagnostics] = []
    expected_requirement_total = 0
    matched_requirement_total = 0
    incomplete_count = 0
    downstream_values: list[bool] = []

    for case in cases:
        diagnostic = decomposer.decompose_with_diagnostics(case.user_query)
        diagnostics.append(diagnostic)

        if case.downstream_success is not None:
            downstream_values.append(case.downstream_success)

        expected_requirements = _expected_requirement_pairs(case)
        if expected_requirements:
            expected_requirement_total += len(expected_requirements)
            actual_requirements = _actual_requirement_pairs(diagnostic.task)
            matched_requirement_total += len(expected_requirements & actual_requirements)
            if not expected_requirements.issubset(actual_requirements):
                incomplete_count += 1
        elif not diagnostic.is_complete:
            incomplete_count += 1

    valid_json_count = sum(1 for diagnostic in diagnostics if diagnostic.valid_json)
    requirement_accuracy = (
        matched_requirement_total / expected_requirement_total
        if expected_requirement_total
        else None
    )
    downstream_success = (
        sum(1 for value in downstream_values if value) / len(downstream_values)
        if downstream_values
        else None
    )

    return DecompositionEvaluationReport(
        total_cases=len(cases),
        valid_json_rate=valid_json_count / len(cases),
        requirement_extraction_accuracy=requirement_accuracy,
        incomplete_decomposition_rate=incomplete_count / len(cases),
        downstream_task_success=downstream_success,
        diagnostics=diagnostics,
    )


def _build_decomposition_prompt(user_query: str) -> str:
    return f"""
You are EcoBudget's task decomposer. Convert the user's research request into
strict JSON only. Do not answer the user's question.

Return exactly this JSON object shape:
{{
  "user_query": "same cleaned user query",
  "task_type": "comparison | specification_lookup | recommendation | multi_source_research | travel_product_decision | information_task",
  "entities": ["specific things, people, places, products, or concepts to research"],
  "attributes": ["specific facts or angles that must be retrieved"],
  "requirements": [
    {{
      "id": "lowercase_snake_case_unique_id",
      "entity": "one value from entities",
      "attribute": "one value from attributes",
      "constraints": {{}}
    }}
  ],
  "answer_format": "short_answer | specification_summary | evidence_backed_summary | comparison_table_with_summary | recommendation_with_evidence",
  "constraints": {{}},
  "warnings": []
}}

Rules:
- Output JSON only, with no markdown.
- Do not include facts or final answers.
- Keep entities and attributes concise.
- Generate one requirement for every entity and attribute combination.
- The requirements list length must equal len(entities) * len(attributes).
- Every requirement attribute must exactly match one value from attributes.
- Every requirement entity must exactly match one value from entities.
- Put dates, locations, budgets, freshness needs, and user preferences in constraints.
- For current-time requests, include a freshness constraint such as "right now" or "current".
- Do not split freshness words such as "current", "latest", "today", or "right now" into separate attributes.
- Do not put requested facts such as weather, price, overview, definition, release date, rating, or availability in entities; those belong in attributes.
- In a query like "weather in Bolivia", the entity is "Bolivia" and the attribute is "current weather".
- In a query like "price of Thar", the entity is "Thar" and the attribute is "current price".
- If the request is broad but answerable, use one entity and an attribute such as "overview".
- If the request is ambiguous, still decompose it and add a warning.

Examples:
- "what is the weather in bolivia right now?" -> entity "Bolivia", attribute "current weather", constraints {{"freshness": "right now"}}, one requirement.
- "what is percy jackson?" -> entity "Percy Jackson", attribute "overview", one requirement.
- "what is the price of thar?" -> entity "Thar", attribute "current price", constraints {{"freshness": "current"}}, one requirement, warning if the entity is ambiguous.

User query: {json.dumps(user_query)}
""".strip()


def _build_repair_prompt(
    user_query: str,
    *,
    raw_response: str,
    validation_errors: list[str],
) -> str:
    return f"""
You are EcoBudget's task decomposer. Your previous response failed validation.
Repair it and return strict JSON only. Do not answer the user's question.

User query: {json.dumps(user_query)}

Validation errors:
{json.dumps(validation_errors, indent=2)}

Previous response:
{raw_response}

Return the same JSON object shape as before. The fixed JSON must satisfy:
- user_query, task_type, entities, attributes, requirements, and answer_format are present.
- entities and attributes are non-empty arrays of strings.
- requirements length equals len(entities) * len(attributes).
- every requirement entity exactly matches one entity.
- every requirement attribute exactly matches one attribute.
- every requirement has a unique lowercase snake_case id.
- requested facts such as weather, price, overview, definition, release date, rating, or availability are attributes, not entities.
- for "weather in Bolivia", use entity "Bolivia" and attribute "current weather".
- output JSON only, with no markdown.
""".strip()


def _task_from_payload(payload: Any, *, expected_query: str) -> DecomposedTask:
    if not isinstance(payload, dict):
        raise TaskDecompositionError("model JSON root must be an object")

    task_type = _required_string(payload, "task_type")
    entities = _required_string_list(payload, "entities")
    attributes = _required_string_list(payload, "attributes")
    answer_format = _required_string(payload, "answer_format")
    constraints = _optional_dict(payload, "constraints")
    warnings = _optional_string_list(payload, "warnings")
    requirement_payloads = payload.get("requirements")
    if not isinstance(requirement_payloads, list):
        raise TaskDecompositionError("requirements must be a list")

    requirements: list[Requirement] = []
    for index, item in enumerate(requirement_payloads, start=1):
        if not isinstance(item, dict):
            raise TaskDecompositionError(f"requirement {index} must be an object")
        requirements.append(
            Requirement(
                id=_required_string(item, "id"),
                entity=_required_string(item, "entity"),
                attribute=_required_string(item, "attribute"),
                constraints=_optional_dict(item, "constraints"),
            )
        )
    constraints = _merge_shared_constraints(constraints, requirements)
    requirements = [
        Requirement(
            id=requirement.id,
            entity=requirement.entity,
            attribute=requirement.attribute,
            constraints={**constraints, **requirement.constraints},
        )
        for requirement in requirements
    ]

    return DecomposedTask(
        user_query=_clean_space(str(payload.get("user_query") or expected_query)),
        task_type=task_type,
        entities=entities,
        attributes=attributes,
        requirements=requirements,
        answer_format=answer_format,
        constraints=constraints,
        warnings=warnings,
    )


def _merge_shared_constraints(
    constraints: dict[str, Any],
    requirements: list[Requirement],
) -> dict[str, Any]:
    merged = dict(constraints)
    if not requirements:
        return merged

    common_keys = set(requirements[0].constraints)
    for requirement in requirements[1:]:
        common_keys &= set(requirement.constraints)

    for key in common_keys:
        values = [requirement.constraints[key] for requirement in requirements]
        if all(value == values[0] for value in values) and key not in merged:
            merged[key] = values[0]
    return merged


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise json.JSONDecodeError("JSON root is not an object", cleaned, 0)
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskDecompositionError(f"{key} must be a non-empty string")
    return _clean_space(value)


def _required_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise TaskDecompositionError(f"{key} must be a non-empty list")

    cleaned: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise TaskDecompositionError(f"{key}[{index}] must be a non-empty string")
        cleaned.append(_clean_space(item))
    return _dedupe(cleaned)


def _optional_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise TaskDecompositionError(f"{key} must be a list")
    cleaned: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            raise TaskDecompositionError(f"{key}[{index}] must be a string")
        item = _clean_space(item)
        if item:
            cleaned.append(item)
    return cleaned


def _optional_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TaskDecompositionError(f"{key} must be an object")
    return value


def _expected_requirement_pairs(case: EvaluationCase) -> set[tuple[str, str]]:
    if case.expected_requirements:
        return {
            (_normalize_for_match(entity), _normalize_for_match(attribute))
            for entity, attribute in case.expected_requirements
        }
    if case.expected_entities and case.expected_attributes:
        return {
            (_normalize_for_match(entity), _normalize_for_match(attribute))
            for entity in case.expected_entities
            for attribute in case.expected_attributes
        }
    return set()


def _actual_requirement_pairs(task: DecomposedTask | None) -> set[tuple[str, str]]:
    if task is None:
        return set()
    return {
        (_normalize_for_match(requirement.entity), _normalize_for_match(requirement.attribute))
        for requirement in task.requirements
    }


def _clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_code_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return slug or "requirement"


def _normalize_for_match(value: str) -> str:
    return _slugify(_clean_space(value))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose an EcoBudget research task.")
    parser.add_argument("query", help="User research task to decompose.")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model tag to use.")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama HTTP host.")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print validation diagnostics instead of raising on invalid model output.",
    )
    args = parser.parse_args()

    decomposer = OllamaTaskDecomposer(model=args.model, host=args.host)
    if args.diagnostics:
        print(json.dumps(decomposer.decompose_with_diagnostics(args.query).to_dict(), indent=2))
        return

    print(decomposer.decompose(args.query).to_json())


if __name__ == "__main__":
    main()
