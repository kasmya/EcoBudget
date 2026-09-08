import json
import unittest

from ecobudget.task_decomposer import (
    EvaluationCase,
    OllamaTaskDecomposer,
    TaskDecompositionError,
    decompose_task,
    evaluate_task_decomposer,
)


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No fake model response queued.")
        response = self.responses.pop(0)
        return response if isinstance(response, str) else json.dumps(response)


def task_payload(
    *,
    user_query="Compare iPhone 15 and Galaxy S24 based on price and battery capacity.",
    task_type="comparison",
    entities=None,
    attributes=None,
    requirements=None,
    answer_format="comparison_table_with_summary",
    constraints=None,
    warnings=None,
):
    entities = entities or ["iPhone 15", "Galaxy S24"]
    attributes = attributes or ["price", "battery capacity"]
    if requirements is None:
        requirements = [
            {
                "id": "iphone_15_price",
                "entity": "iPhone 15",
                "attribute": "price",
                "constraints": {},
            },
            {
                "id": "iphone_15_battery_capacity",
                "entity": "iPhone 15",
                "attribute": "battery capacity",
                "constraints": {},
            },
            {
                "id": "galaxy_s24_price",
                "entity": "Galaxy S24",
                "attribute": "price",
                "constraints": {},
            },
            {
                "id": "galaxy_s24_battery_capacity",
                "entity": "Galaxy S24",
                "attribute": "battery capacity",
                "constraints": {},
            },
        ]
    return {
        "user_query": user_query,
        "task_type": task_type,
        "entities": entities,
        "attributes": attributes,
        "requirements": requirements,
        "answer_format": answer_format,
        "constraints": constraints or {},
        "warnings": warnings or [],
    }


class TaskDecomposerTests(unittest.TestCase):
    def test_model_output_generates_entity_attribute_grid(self):
        client = FakeModelClient([task_payload()])
        decomposer = OllamaTaskDecomposer(client=client)

        task = decomposer.decompose(
            "Compare iPhone 15 and Galaxy S24 based on price and battery capacity."
        )

        self.assertEqual(task.task_type, "comparison")
        self.assertEqual(task.entities, ["iPhone 15", "Galaxy S24"])
        self.assertEqual(task.attributes, ["price", "battery capacity"])
        self.assertEqual(len(task.requirements), 4)
        self.assertEqual(task.answer_format, "comparison_table_with_summary")

    def test_model_handles_current_weather_query_shape(self):
        client = FakeModelClient(
            [
                task_payload(
                    user_query="what is the weather in bolivia right now?",
                    task_type="specification_lookup",
                    entities=["Bolivia"],
                    attributes=["current weather"],
                    requirements=[
                        {
                            "id": "bolivia_current_weather",
                            "entity": "Bolivia",
                            "attribute": "current weather",
                            "constraints": {"freshness": "right now"},
                        }
                    ],
                    answer_format="short_answer",
                    constraints={"freshness": "right now"},
                )
            ]
        )
        decomposer = OllamaTaskDecomposer(client=client)

        task = decomposer.decompose("what is the weather in bolivia right now?")

        self.assertEqual(task.entities, ["Bolivia"])
        self.assertEqual(task.attributes, ["current weather"])
        self.assertEqual(task.constraints["freshness"], "right now")

    def test_model_handles_general_information_query_shape(self):
        client = FakeModelClient(
            [
                task_payload(
                    user_query="what is percy jackson?",
                    task_type="information_task",
                    entities=["Percy Jackson"],
                    attributes=["overview"],
                    requirements=[
                        {
                            "id": "percy_jackson_overview",
                            "entity": "Percy Jackson",
                            "attribute": "overview",
                            "constraints": {},
                        }
                    ],
                    answer_format="short_answer",
                )
            ]
        )
        decomposer = OllamaTaskDecomposer(client=client)

        task = decomposer.decompose("what is percy jackson?")

        self.assertEqual(task.task_type, "information_task")
        self.assertEqual(task.entities, ["Percy Jackson"])
        self.assertEqual(task.attributes, ["overview"])

    def test_model_handles_price_lookup_query_shape(self):
        client = FakeModelClient(
            [
                task_payload(
                    user_query="what is the price of thar?",
                    task_type="specification_lookup",
                    entities=["Thar"],
                    attributes=["current price"],
                    requirements=[
                        {
                            "id": "thar_current_price",
                            "entity": "Thar",
                            "attribute": "current price",
                            "constraints": {"freshness": "current"},
                        }
                    ],
                    answer_format="short_answer",
                    constraints={"freshness": "current"},
                    warnings=["Entity may refer to Mahindra Thar depending on user context."],
                )
            ]
        )
        decomposer = OllamaTaskDecomposer(client=client)

        task = decomposer.decompose("what is the price of thar?")

        self.assertEqual(task.entities, ["Thar"])
        self.assertEqual(task.attributes, ["current price"])
        self.assertIn("Entity may refer", task.warnings[0])

    def test_invalid_json_fails(self):
        decomposer = OllamaTaskDecomposer(
            client=FakeModelClient(["not json", "not json"]),
        )

        with self.assertRaises(TaskDecompositionError):
            decomposer.decompose("what is percy jackson?")

        diagnostic = OllamaTaskDecomposer(
            client=FakeModelClient(["not json"]),
            max_attempts=1,
        ).decompose_with_diagnostics("what is percy jackson?")
        self.assertFalse(diagnostic.valid_json)

    def test_incomplete_decomposition_fails_validation(self):
        incomplete_payload = task_payload(requirements=[])
        decomposer = OllamaTaskDecomposer(
            client=FakeModelClient([incomplete_payload]),
            max_attempts=1,
        )

        with self.assertRaises(TaskDecompositionError) as context:
            decomposer.decompose("Compare iPhone 15 and Galaxy S24 based on price.")

        self.assertIn("no requirements generated", str(context.exception))

    def test_repair_attempt_can_fix_incomplete_decomposition(self):
        incomplete_payload = task_payload(requirements=[])
        repaired_payload = task_payload()
        client = FakeModelClient([incomplete_payload, repaired_payload])
        decomposer = OllamaTaskDecomposer(client=client, max_attempts=2)

        task = decomposer.decompose(
            "Compare iPhone 15 and Galaxy S24 based on price and battery capacity."
        )

        self.assertEqual(len(client.prompts), 2)
        self.assertIn("failed validation", client.prompts[1])
        self.assertEqual(len(task.requirements), 4)

    def test_empty_query_fails_before_model_call(self):
        with self.assertRaises(TaskDecompositionError):
            decompose_task("   ")

    def test_evaluation_report_metrics(self):
        good_payload = task_payload()
        incomplete_payload = task_payload(requirements=[])
        decomposer = OllamaTaskDecomposer(
            client=FakeModelClient([good_payload, "not json", incomplete_payload]),
            max_attempts=1,
        )

        report = evaluate_task_decomposer(
            decomposer,
            [
                EvaluationCase(
                    user_query="Compare iPhone 15 and Galaxy S24 based on price and battery capacity.",
                    expected_entities=["iPhone 15", "Galaxy S24"],
                    expected_attributes=["price", "battery capacity"],
                    downstream_success=True,
                ),
                EvaluationCase(
                    user_query="what is percy jackson?",
                    expected_entities=["Percy Jackson"],
                    expected_attributes=["overview"],
                    downstream_success=False,
                ),
                EvaluationCase(
                    user_query="Compare iPhone 15 and Galaxy S24 based on price.",
                    expected_entities=["iPhone 15", "Galaxy S24"],
                    expected_attributes=["price"],
                ),
            ],
        )

        self.assertEqual(report.total_cases, 3)
        self.assertAlmostEqual(report.valid_json_rate, 2 / 3)
        self.assertAlmostEqual(report.requirement_extraction_accuracy, 4 / 7)
        self.assertAlmostEqual(report.incomplete_decomposition_rate, 2 / 3)
        self.assertAlmostEqual(report.downstream_task_success, 1 / 2)


if __name__ == "__main__":
    unittest.main()
