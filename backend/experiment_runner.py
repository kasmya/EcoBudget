import json
import time
import uuid
import csv
import os
from datetime import datetime, timezone

from scorer import parse_resources, check_task_success_v2
from conditions import run_normal, run_fixed_eco, run_ecobudget

BENCHMARK_FILE = "benchmark_tasks.json"
RESULTS_DIR = "results"
CONDITIONS = ["normal", "fixed_eco", "ecobudget"]


def load_benchmark():
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


def run_condition(name, task_text, resources):
    if name == "normal":
        return run_normal(task_text, resources)
    if name == "fixed_eco":
        return run_fixed_eco(task_text, resources)
    if name == "ecobudget":
        return run_ecobudget(task_text, resources)
    raise ValueError(f"unknown condition {name}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"run_{timestamp}_{run_id}.csv")

    tasks = load_benchmark()

    fieldnames = [
        "run_id", "task_id", "category", "difficulty", "condition",
        "bytes_used", "resources_loaded", "resources_total",
        "budget_final", "iterations", "time_seconds",
        "extracted_answer", "answer_score",
        "facts_matched", "facts_total", "success",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            print(f"\n=== {task['id']}: {task['question']} ===")
            with open(task["page_file"]) as pf:
                html = pf.read()
            resources = parse_resources(html, base_url=task["url"])

            for cond in CONDITIONS:
                start = time.time()
                result = run_condition(cond, task["question"], resources)
                elapsed = time.time() - start

                success, matched, total = check_task_success_v2(
                    result["answer"], result["selected_text"], task["ground_truth"]
                )

                row = {
                    "run_id": run_id,
                    "task_id": task["id"],
                    "category": task["category"],
                    "difficulty": task["difficulty"],
                    "condition": result["condition"],
                    "bytes_used": result["bytes_used"],
                    "resources_loaded": result["resources_loaded"],
                    "resources_total": result["resources_total"],
                    "budget_final": result["budget_final"],
                    "iterations": result["iterations"],
                    "time_seconds": round(elapsed, 3),
                    "extracted_answer": result["answer"],
                    "answer_score": round(result["answer_score"], 3),
                    "facts_matched": matched,
                    "facts_total": total,
                    "success": success,
                }
                writer.writerow(row)
                print(f"  {result['condition']:10s} | "
                      f"{result['bytes_used']:>7} bytes (budget={result['budget_final']}) | "
                      f"success={success} ({matched}/{total}) | answer='{result['answer']}'")

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
