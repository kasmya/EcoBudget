"""Live EcoBudget-vs-normal demo server (dependency-free stdlib).

Serves the light-themed dashboard and a /api/compare endpoint that:
  1. calls Tavily (server-side; key from the TAVILY_API_KEY env var) to fetch
     real web results for a query, with full raw page content and snippets;
  2. computes NORMAL loading (transfer every full page) vs ECOBUDGET
     task-sufficient loading (transfer only the relevant snippet that answers),
     using the project's measured 5G energy model (ml_retriever/energy.py);
  3. returns the byte, transfer-energy, and 5G RRC radio-energy comparison.

The key is read from the environment and never written to disk or returned to
the client. Run:

    export TAVILY_API_KEY=<your key>
    python backend/demo_server.py            # serves the whole site on :8770

Then open http://localhost:8770/ (landing page); the live comparison is at
http://localhost:8770/dashboard.html ("Live demo" in the nav).
"""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import mimetypes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml_retriever"))  # import the energy model
from ml_retriever.energy import FiveGTransferModel, RadioStateModel  # noqa: E402

SITE = ROOT  # the site root: index.html, dashboard.html, public/ all live here
TAVILY_URL = "https://api.tavily.com/search"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Sets vars not already in the env.
    The .env is gitignored; the key is never logged or served to clients."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(ROOT / ".env")
GRID_G_PER_KWH = 475.0

_transfer = FiveGTransferModel()
_radio_tight = RadioStateModel(demote_between_fetches=False)
_radio_fastd = RadioStateModel(demote_between_fetches=True)


def _energy(num_bytes: int, n_fetches: int) -> dict:
    tj = _transfer.energy_joules(num_bytes)
    rt = _radio_tight.account(num_bytes, n_fetches)
    rf = _radio_fastd.account(num_bytes, n_fetches)
    total = tj + rt["radio_j"]
    return {
        "bytes": int(num_bytes),
        "fetches": int(n_fetches),
        "transfer_j": round(tj, 4),
        "radio_j_tight": round(rt["radio_j"], 2),
        "radio_j_fastdormancy": round(rf["radio_j"], 2),
        "radio_active_s": round(rt["radio_active_time_s"], 2),
        "co2e_mg": round((total / 3.6e6) * GRID_G_PER_KWH * 1000, 4),
    }


def tavily_search(query: str, key: str, max_results: int = 4) -> dict:
    body = json.dumps({"query": query, "include_raw_content": True,
                       "include_answer": True, "max_results": max_results}).encode()
    req = urllib.request.Request(TAVILY_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


MEASURED_HTML_PAGE = 506_179   # measured median real HTML page (data/measured_payloads.json)
FULL_PAGE = 2_000_000          # full page with assets (HTTP-Archive-class)


def compare(query: str, key: str) -> dict:
    r = tavily_search(query, key)
    answer = (r.get("answer") or "").strip()
    results = r.get("results", []) or []
    sources = []
    text_bytes = 0
    for x in results:
        raw = (x.get("raw_content") or "")
        snip = (x.get("content") or "")
        rb = len(raw.encode("utf-8"))
        text_bytes += rb
        sources.append({"url": x.get("url", ""), "raw_bytes": rb,
                        "snippet_bytes": len(snip.encode("utf-8"))})
    n_pages = max(1, len(results))
    # EcoBudget task-sufficient payload: only the relevant snippet that answers.
    best_snip = results[0].get("content", "") if results else ""
    eco_bytes = len((best_snip or answer).encode("utf-8")) or 1

    def side(nbytes, nfetch, ebytes, efetch):
        n = _energy(nbytes, nfetch)
        e = _energy(ebytes, efetch)
        pct = round(100 * (1 - ebytes / nbytes), 1) if nbytes else 0.0
        return {
            "normal": n, "ecobudget": e,
            "savings": {
                "bytes_pct": pct,
                "bytes": nbytes - ebytes,
                "transfer_j": round(n["transfer_j"] - e["transfer_j"], 4),
                "radio_j_fastdormancy": round(n["radio_j_fastdormancy"] - e["radio_j_fastdormancy"], 2),
                "radio_j_tight": round(n["radio_j_tight"] - e["radio_j_tight"], 2),
                "co2e_mg": round(n["co2e_mg"] - e["co2e_mg"], 4),
                "radio_active_s": round(n["radio_active_s"] - e["radio_active_s"], 2),
            },
        }

    scenarios = {
        # what a browser/agent actually ingested (real measured bytes)
        "text": side(text_bytes, n_pages, eco_bytes, 1),
        # realistic full HTML documents (measured 506 KB median)
        "html_page": side(n_pages * MEASURED_HTML_PAGE, n_pages, eco_bytes, 1),
        # full pages with assets
        "full_page": side(n_pages * FULL_PAGE, n_pages, eco_bytes, 1),
    }
    return {
        "query": query,
        "answer": answer,
        "n_results": len(results),
        "n_pages": n_pages,
        "eco_bytes": eco_bytes,
        "sources": sources,
        "scenarios": scenarios,
        "default_scenario": "html_page",
        "note": ("Normal loads every result page; EcoBudget loads only the task-relevant "
                 "snippet. 'text' uses the real fetched bytes; 'html_page' (506 KB measured "
                 "median) and 'full_page' (2 MB) are realistic page weights. Energy uses the "
                 "project's 5G per-bit transfer + RRC radio-state model."),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/api/compare":
            key = os.environ.get("TAVILY_API_KEY", "").strip()
            if not key:
                return self._send(500, json.dumps({"error": "TAVILY_API_KEY not set on the server"}))
            q = (parse_qs(p.query).get("q", [""])[0]).strip()
            if not q:
                return self._send(400, json.dumps({"error": "missing query ?q="}))
            try:
                return self._send(200, json.dumps(compare(q, key)))
            except Exception as e:
                return self._send(502, json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"}))
        # static site: / -> index.html; only web assets served, from the repo root
        rel = "index.html" if p.path in ("/", "") else p.path.lstrip("/")
        candidate = (SITE / rel).resolve()
        allowed_ext = {".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg",
                       ".gif", ".webp", ".ico", ".mp4", ".webm", ".json", ".woff2"}
        if (SITE in candidate.parents and candidate.is_file()
                and candidate.suffix.lower() in allowed_ext):
            ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            return self._send(200, candidate.read_bytes(), ctype)
        return self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *a):  # quieter
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8770"))
    if not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY is not set; /api/compare will return an error until you export it.")
    print(f"EcoBudget demo server on http://localhost:{port}/  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
