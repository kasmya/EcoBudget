# EcoBudget

A research prototype for task-sufficient web loading: given a question and an
HTML page, it ranks passages by likely usefulness per byte, compares
full-page / fixed-budget / adaptive loading strategies, and estimates the
transfer-related CO2e of each. See [codebase-analysis.md](codebase-analysis.md)
for a full architectural walkthrough of the existing prototype.

## Layout

- **`backend/` + `frontend/`** — the original FastAPI + static-HTML prototype
  (demo API, experiment runner, CO2e estimation). No package manager
  manifest yet; dependencies are implicit in the imports (FastAPI,
  BeautifulSoup, sentence-transformers, Playwright). Fixture pages under
  `backend/pages/` are not tracked in git — regenerate them with
  `backend/download_pages.sh` or `backend/fetch_missing_pages.py`.
- **`ml_retriever/`** — the active ML/retrieval workstream (task
  decomposition, requirement-aware retrieval, evidence tracking, adaptive
  policy, answer generation), built local-models-only. See
  [`ml_retriever/README.md`](ml_retriever/README.md) for setup and tests.
- **[`plan.md`](plan.md)** — the phased roadmap for the `ml_retriever`
  workstream, including what's reused from `backend/` vs. built new.

## Getting started

```bash
cd ml_retriever
pip install -e ".[dev]"
python -m pytest -v
```

For the legacy `backend/` demo, install its imports manually (no
`requirements.txt` yet — see `codebase-analysis.md` for the full dependency
list) and run `uvicorn main:app --reload` from `backend/`.
