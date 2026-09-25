"""EcoBudget backend (FastAPI) wired to the seeded SQLite database.

Every number the API returns is computed from real data in data/ecobudget.db:
emission factors from DEFRA + Our World in Data, benchmarks from OWID, and the
user's own logged entries. kg CO2e = quantity x published factor. No mocked or
hardcoded results.

Run: uvicorn main:app --reload  (from the ecobudget_app directory)
"""
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "ecobudget.db"

app = FastAPI(title="EcoBudget", version="1.0",
              description="Personal carbon budget tracker on real emission-factor data.")


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def this_month() -> str:
    return date.today().strftime("%Y-%m")


class NewEntry(BaseModel):
    activity: str
    quantity: float = Field(gt=0)
    entry_date: Optional[str] = None   # YYYY-MM-DD, defaults to today
    note: Optional[str] = None


class BudgetUpdate(BaseModel):
    monthly_kg_co2e: float = Field(gt=0)


# ---------------------------------------------------------------- factors
@app.get("/api/factors")
def factors():
    con = db()
    rows = con.execute(
        "SELECT id, category, activity, unit, kg_co2e_per_unit, source, year "
        "FROM emission_factors ORDER BY category, kg_co2e_per_unit DESC").fetchall()
    con.close()
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["category"], []).append(dict(r))
    return {"categories": list(grouped.keys()), "factors": grouped, "count": len(rows)}


# ---------------------------------------------------------------- entries
@app.get("/api/entries")
def list_entries(month: Optional[str] = None):
    month = month or this_month()
    con = db()
    rows = con.execute(
        "SELECT * FROM entries WHERE substr(entry_date,1,7)=? ORDER BY entry_date DESC, id DESC",
        (month,)).fetchall()
    con.close()
    return {"month": month, "entries": [dict(r) for r in rows]}


@app.post("/api/entries", status_code=201)
def add_entry(e: NewEntry):
    con = db()
    f = con.execute(
        "SELECT id, category, unit, kg_co2e_per_unit FROM emission_factors WHERE activity=?",
        (e.activity,)).fetchone()
    if not f:
        con.close()
        raise HTTPException(404, f"Unknown activity: {e.activity!r}")
    entry_date = e.entry_date or date.today().isoformat()
    kg = round(e.quantity * f["kg_co2e_per_unit"], 3)
    cur = con.execute(
        "INSERT INTO entries (created_ts, entry_date, factor_id, category, activity, unit, quantity, kg_co2e, note)"
        " VALUES (datetime('now'),?,?,?,?,?,?,?,?)",
        (entry_date, f["id"], f["category"], e.activity, f["unit"], e.quantity, kg, e.note),
    )
    con.commit()
    row = con.execute("SELECT * FROM entries WHERE id=?", (cur.lastrowid,)).fetchone()
    con.close()
    return dict(row)


@app.delete("/api/entries/{entry_id}", status_code=204)
def delete_entry(entry_id: int):
    con = db()
    cur = con.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    con.commit()
    con.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Entry not found")


# ---------------------------------------------------------------- summary
@app.get("/api/summary")
def summary(month: Optional[str] = None):
    month = month or this_month()
    con = db()
    b = con.execute("SELECT monthly_kg_co2e, annual_kg_co2e, basis FROM budget WHERE id=1").fetchone()
    budget_kg = b["monthly_kg_co2e"]

    total = con.execute(
        "SELECT COALESCE(SUM(kg_co2e),0) AS t, COUNT(*) AS n FROM entries WHERE substr(entry_date,1,7)=?",
        (month,)).fetchone()
    total_kg = round(total["t"], 2)

    by_cat = con.execute(
        "SELECT category, ROUND(SUM(kg_co2e),2) AS kg, COUNT(*) AS n "
        "FROM entries WHERE substr(entry_date,1,7)=? GROUP BY category ORDER BY kg DESC",
        (month,)).fetchall()
    by_day = con.execute(
        "SELECT entry_date AS day, ROUND(SUM(kg_co2e),2) AS kg "
        "FROM entries WHERE substr(entry_date,1,7)=? GROUP BY entry_date ORDER BY entry_date",
        (month,)).fetchall()
    top = con.execute(
        "SELECT activity, category, ROUND(SUM(kg_co2e),2) AS kg FROM entries "
        "WHERE substr(entry_date,1,7)=? GROUP BY activity ORDER BY kg DESC LIMIT 5",
        (month,)).fetchall()
    con.close()

    remaining = round(budget_kg - total_kg, 2)
    pct = round(100 * total_kg / budget_kg, 1) if budget_kg else 0
    return {
        "month": month,
        "total_kg": total_kg,
        "budget_kg": budget_kg,
        "remaining_kg": remaining,
        "pct_of_budget": pct,
        "over_budget": total_kg > budget_kg,
        "entry_count": total["n"],
        "budget_basis": b["basis"],
        "by_category": [dict(r) for r in by_cat],
        "by_day": [dict(r) for r in by_day],
        "top_activities": [dict(r) for r in top],
    }


# ---------------------------------------------------------------- benchmarks
@app.get("/api/benchmarks")
def benchmarks(month: Optional[str] = None):
    month = month or this_month()
    con = db()
    rows = con.execute(
        "SELECT country, year, co2_per_capita_tonnes, consumption_co2_per_capita_tonnes "
        "FROM country_benchmarks ORDER BY co2_per_capita_tonnes DESC").fetchall()
    total = con.execute(
        "SELECT COALESCE(SUM(kg_co2e),0) AS t FROM entries WHERE substr(entry_date,1,7)=?",
        (month,)).fetchone()["t"]
    con.close()
    # Project an annual footprint from the month so far: average per elapsed day x 365.
    day_of_month = date.today().day if month == this_month() else 30
    daily_avg = (total / day_of_month) if day_of_month else 0
    projected_annual_t = round(daily_avg * 365 / 1000, 2)
    return {
        "source": "Our World in Data (CC-BY 4.0), annual CO2 per capita, latest year",
        "your_projected_annual_tonnes": projected_annual_t,
        "note": "Projected from this month's logged entries (daily average x 365). "
                "Logs are a personal subset, so this is a partial-footprint estimate.",
        "countries": [dict(r) for r in rows],
    }


# ---------------------------------------------------------------- budget
@app.get("/api/budget")
def get_budget():
    con = db()
    b = con.execute("SELECT monthly_kg_co2e, annual_kg_co2e, basis FROM budget WHERE id=1").fetchone()
    con.close()
    return dict(b)


@app.put("/api/budget")
def set_budget(u: BudgetUpdate):
    con = db()
    con.execute("UPDATE budget SET monthly_kg_co2e=?, annual_kg_co2e=? WHERE id=1",
                (u.monthly_kg_co2e, round(u.monthly_kg_co2e * 12, 1)))
    con.commit()
    b = con.execute("SELECT monthly_kg_co2e, annual_kg_co2e, basis FROM budget WHERE id=1").fetchone()
    con.close()
    return dict(b)


# ---------------------------------------------------------------- static UI
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/", StaticFiles(directory=ROOT / "static"), name="static")
