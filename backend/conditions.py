"""
Three loading conditions compared in the EcoBudget experiment.
All three consume the same `resources` list produced by scorer.parse_resources().

IMPORTANT: we never concatenate multiple passages into one string before
asking the QA model — that confuses extraction and returns None even when
the answer is present in one of the individual passages. Instead we score
each passage separately and keep the best-scoring one.
"""

from scorer import rank_by_vpb, find_best_answer_full_page


def select_within_budget(ranked_resources, budget_bytes):
    """Greedy: only add a resource if it fits in the remaining budget.
    Guarantees used_bytes never exceeds budget_bytes."""
    selected = []
    used_bytes = 0
    for r in ranked_resources:
        if used_bytes + r['bytes'] <= budget_bytes:
            selected.append(r)
            used_bytes += r['bytes']
    return selected, used_bytes


def best_from_selected(selected):
    """Pick the best-scoring resource from an already-ranked, already-scored
    subset (rank_by_vpb sets 'utility' and 'extracted_answer' per resource)."""
    if not selected:
        return 0.0, None
    best = max(selected, key=lambda r: r['utility'])
    return best['utility'], best['extracted_answer']


def run_normal(task_text, resources):
    """Baseline: load the entire page. Scored per-passage (not concatenated)
    so this is a fair comparison, not artificially handicapped."""
    text_resources = [r for r in resources if r['type'] == 'text']
    total_bytes = sum(r['bytes'] for r in resources)
    text_blob = " ".join(r['content'] for r in text_resources)  # for fact-matching only

    score, answer = find_best_answer_full_page(task_text, resources, use_entity_filter=False)
    return {
        "condition": "Normal",
        "bytes_used": total_bytes,
        "resources_loaded": len(resources),
        "resources_total": len(resources),
        "answer": answer,
        "answer_score": score,
        "selected_text": text_blob,
        "budget_final": None,
        "iterations": 1,
    }


def run_fixed_eco(task_text, resources, fixed_budget_bytes=20_000):
    """Fixed rule: top resources by embedding similarity within a fixed byte budget."""
    text_resources = [r for r in resources if r['type'] == 'text']
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=False)

    selected, used_bytes = select_within_budget(ranked, fixed_budget_bytes)
    score, answer = best_from_selected(selected)
    text_blob = " ".join(r['content'] for r in selected)

    return {
        "condition": "Fixed Eco",
        "bytes_used": used_bytes,
        "resources_loaded": len(selected),
        "resources_total": len(text_resources),
        "answer": answer,
        "answer_score": score,
        "selected_text": text_blob,
        "budget_final": fixed_budget_bytes,
        "iterations": 1,
    }


def run_ecobudget(task_text, resources, start_budget=10_000,
                   growth_factor=2.0, max_iterations=5,
                   success_threshold=0.3):
    """Adaptive: start small, check answerability, grow budget only if needed."""
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=True)

    budget = start_budget
    iterations = 0
    best_score = 0.0
    best_answer = None
    best_text = ""
    used_bytes = 0
    n_loaded = 0

    for iterations in range(1, max_iterations + 1):
        selected, used_bytes = select_within_budget(ranked, budget)
        n_loaded = len(selected)
        score, answer = best_from_selected(selected)
        text_blob = " ".join(r['content'] for r in selected)

        if score > best_score:
            best_score = score
            best_answer = answer
            best_text = text_blob

        if score >= success_threshold:
            break
        if used_bytes >= sum(r['bytes'] for r in ranked):
            break

        budget = int(budget * growth_factor)

    return {
        "condition": "EcoBudget",
        "bytes_used": used_bytes,
        "resources_loaded": n_loaded,
        "resources_total": len(ranked),
        "answer": best_answer,
        "answer_score": best_score,
        "selected_text": best_text,
        "budget_final": budget,
        "iterations": iterations,
    }
