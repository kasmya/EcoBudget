from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scorer import parse_resources, rank_by_vpb, check_task_success, find_best_answer_full_page
from db import init_db, log_decision, get_all_logs

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()

SUFFICIENCY_THRESHOLD = 0.55
FIXED_BUDGET_BYTES = 2_000_000


class RunRequest(BaseModel):
    task_text: str
    html: str
    participant_id: str
    task_id: str
    condition: str  # "normal", "fixed", "ecobudget"
    ground_truth: str = ""


@app.post("/run")
def run_ecobudget(req: RunRequest):
    resources = parse_resources(req.html)

    delivered = []
    total_bytes = 0
    confidence = 0.0
    final_answer = None

    if req.condition == "normal":
        # Genuinely checks the ENTIRE page, not a pre-filtered subset
        all_resources = resources
        delivered = all_resources
        total_bytes = sum(r.get('bytes', 0) for r in all_resources)
        confidence, final_answer = find_best_answer_full_page(req.task_text, resources)

    elif req.condition == "fixed":
        ranked = rank_by_vpb(req.task_text, resources)
        budget = FIXED_BUDGET_BYTES
        for r in ranked:
            if total_bytes + r['bytes'] <= budget:
                delivered.append(r)
                total_bytes += r['bytes']
        if delivered:
            best = max(delivered, key=lambda r: r['utility'])
            confidence = best['utility']
            final_answer = best['extracted_answer']

    else:  # ecobudget
        ranked = rank_by_vpb(req.task_text, resources)
        for r in ranked:
            delivered.append(r)
            total_bytes += r['bytes']

            if r['utility'] > confidence:
                confidence = r['utility']
                final_answer = r['extracted_answer']

            log_decision({
                "participant_id": req.participant_id,
                "task_id": req.task_id,
                "condition": req.condition,
                "resource_type": r['type'],
                "bytes": r['bytes'],
                "utility": r['utility'],
                "vpb": r['vpb'],
                "loaded": 1,
                "task_confidence": confidence,
                "task_success": None,
                "completion_time": None,
            })

            if confidence >= SUFFICIENCY_THRESHOLD:
                break

    task_success = None
    if req.ground_truth:
        task_success = check_task_success(final_answer, req.ground_truth)

    if req.condition in ("normal", "fixed"):
        log_decision({
            "participant_id": req.participant_id,
            "task_id": req.task_id,
            "condition": req.condition,
            "resource_type": "summary",
            "bytes": total_bytes,
            "utility": confidence,
            "vpb": None,
            "loaded": len(delivered),
            "task_confidence": confidence,
            "task_success": int(task_success) if task_success is not None else None,
            "completion_time": None,
        })

    return {
        "condition": req.condition,
        "delivered_count": len(delivered),
        "total_bytes": total_bytes,
        "confidence": confidence,
        "final_answer": final_answer,
        "task_success": task_success,
        "top_resources": [
            {"type": r.get('type'), "content": r.get('content', '')[:100], "bytes": r.get('bytes')}
            for r in delivered[:5]
        ],
    }


@app.get("/logs")
def view_logs():
    return get_all_logs()


@app.get("/health")
def health():
    return {"status": "ok"}
