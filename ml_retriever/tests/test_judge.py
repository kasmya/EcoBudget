import pytest

from ml_retriever.judge import (
    JUDGES,
    NARRATIVE_F1_THRESHOLD,
    PROCEDURE_STEP_F1_THRESHOLD,
    judge_task,
    token_f1,
)


@pytest.mark.parametrize(
    "answer_type",
    ["single_fact", "yes_no", "list", "comparison", "multi_part"],
)
def test_each_structured_judge_has_clear_pass_and_fail(answer_type):
    task = {"answer_type": answer_type, "expected_answer": "New Delhi"}
    assert JUDGES[answer_type]("The answer is New Delhi.", task) == (True, 1.0)
    assert JUDGES[answer_type]("The answer is Tokyo.", task) == (False, 0.0)


def test_procedure_judge_has_clear_pass_and_fail():
    task = {
        "answer_type": "procedure",
        "ground_truth": {
            "required_facts": ["loosen lug nuts", "raise vehicle"],
            "match_threshold": 1.0,
        },
    }
    assert judge_task("Loosen lug nuts then raise vehicle", task) == (True, 1.0)
    assert judge_task("Raise vehicle", task) == (False, 0.5)


def test_narrative_judge_has_clear_pass_and_fail():
    task = {
        "answer_type": "narrative",
        "ground_truth": "Sophie invites three possible fathers to her wedding.",
    }
    success, score = judge_task("Sophie invites three possible fathers to her wedding.", task)
    assert success is True
    assert score == 1.0
    assert judge_task("A phone comparison.", task) == (False, 0.0)


def test_narrative_f1_boundary_is_pinned():
    gold = "one two three four five six seven eight nine ten"
    just_above = "one two three extra"
    just_below = "one two three extra another final"

    assert token_f1(just_above, gold) > NARRATIVE_F1_THRESHOLD
    assert judge_task(
        just_above, {"answer_type": "narrative", "ground_truth": gold}
    )[0] is True
    assert token_f1(just_below, gold) < NARRATIVE_F1_THRESHOLD
    assert judge_task(
        just_below, {"answer_type": "narrative", "ground_truth": gold}
    )[0] is False


def test_procedure_step_f1_boundary_is_pinned():
    step = "one two three four five six seven eight nine ten"
    just_above = "one two three extra"
    just_below = "one two three extra another final"
    task = {
        "answer_type": "procedure",
        "ground_truth": {"required_facts": [step], "match_threshold": 1.0},
    }

    assert token_f1(just_above, step) > PROCEDURE_STEP_F1_THRESHOLD
    assert judge_task(just_above, task) == (True, 1.0)
    assert token_f1(just_below, step) < PROCEDURE_STEP_F1_THRESHOLD
    assert judge_task(just_below, task) == (False, 0.0)
