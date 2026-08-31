"""
EcoBudget v2 conditions.
EcoBudget now uses EvidenceCoverageTracker — it knows WHAT facts the task
requires and stops only when those specific facts have been found.
Ground truth is never used inside any condition.
"""

from scorer import rank_by_vpb, find_best_answer_full_page, check_answerability
from evidence import EvidenceCoverageTracker


def select_within_budget(ranked_resources, budget_bytes):
    selected, used = [], 0
    for r in ranked_resources:
        if used + r['bytes'] <= budget_bytes:
            selected.append(r)
            used += r['bytes']
    return selected, used


def best_from_selected(selected):
    if not selected:
        return 0.0, None
    best = max(selected, key=lambda r: r.get('qa_score', r['utility']))
    return best.get('qa_score', best['utility']), best['extracted_answer']


def run_normal(task_text, resources):
    total_bytes = sum(r['bytes'] for r in resources)
    text_blob = " ".join(r['content'] for r in resources if r['type'] == 'text')
    score, answer = find_best_answer_full_page(task_text, resources, use_entity_filter=False)
    return {
        "condition": "Normal", "bytes_used": total_bytes,
        "resources_loaded": len(resources), "resources_total": len(resources),
        "answer": answer, "answer_score": score, "selected_text": text_blob,
        "budget_final": None, "iterations": 1, "gate_reason": "full page",
    }


def run_fixed_eco(task_text, resources, fixed_budget_bytes=20_000):
    text_resources = [r for r in resources if r['type'] == 'text']
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=False)
    selected, used_bytes = select_within_budget(ranked, fixed_budget_bytes)
    score, answer = best_from_selected(selected)
    return {
        "condition": "Fixed Eco", "bytes_used": used_bytes,
        "resources_loaded": len(selected), "resources_total": len(text_resources),
        "answer": answer, "answer_score": score,
        "selected_text": " ".join(r['content'] for r in selected),
        "budget_final": fixed_budget_bytes, "iterations": 1,
        "gate_reason": "fixed budget",
    }


def run_ecobudget(task_text, resources, max_passages=30, coverage_threshold=0.8):
    """
    Evidence Coverage version: loads passages in VPB order, checks each
    against the specific unfilled requirements for THIS task, stops when
    coverage_threshold fraction of requirements are satisfied.

    Key difference from v1: the tracker knows WHAT facts are needed.
    A passage about Colosseum restoration does NOT satisfy 'when completed'.
    """
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=True)
    text_ranked = [r for r in ranked if r['type'] == 'text']

    tracker = EvidenceCoverageTracker(task_text)

    # QA function wrapper that matches the signature evidence.py expects
    def qa_fn(question, passage):
        score, answer = check_answerability(question, passage)
        return score, answer

    used_bytes = 0
    passages_loaded = 0
    stop_reason = f"max passages ({max_passages})"

    for resource in text_ranked[:max_passages]:
        new_evidence = tracker.check_passage(
            resource['content'], resource['bytes'], qa_fn
        )
        used_bytes += resource['bytes']
        passages_loaded += 1

        if tracker.is_sufficient(threshold=coverage_threshold):
            stop_reason = f"coverage {tracker.coverage():.0%} ({tracker.n_satisfied()}/{tracker.n_total()} requirements)"
            break

    best_answer = tracker.best_answer()

    # If tracker found nothing, fall back to best scored passage
    if best_answer is None:
        score, answer = best_from_selected(text_ranked[:passages_loaded])
        best_answer = answer
        best_score = score
    else:
        satisfied = [r for r in tracker.requirements if r["satisfied"]]
        best_score = max((r.get("confidence", 0) for r in satisfied), default=0.0)

    return {
        "condition": "EcoBudget",
        "bytes_used": used_bytes,
        "resources_loaded": passages_loaded,
        "resources_total": len(text_ranked),
        "answer": best_answer,
        "answer_score": best_score,
        "selected_text": " ".join(
            r['content'] for r in text_ranked[:passages_loaded]
        ),
        "budget_final": used_bytes,
        "iterations": passages_loaded,
        "gate_reason": stop_reason,
        "coverage": tracker.coverage(),
        "requirements_met": tracker.n_satisfied(),
        "requirements_total": tracker.n_total(),
        "coverage_detail": tracker.status_report(),
    }
