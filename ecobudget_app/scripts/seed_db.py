"""Create and seed the EcoBudget SQLite database from the fetched real data.

Tables:
  emission_factors   published kg CO2e per unit, per activity (DEFRA + OWID food)
  country_benchmarks per-capita CO2 by country (OWID, downloaded)
  budget             one row: science-based monthly personal carbon budget
  entries            user activity log; kg_co2e computed as quantity x factor

Run AFTER scripts/fetch_data.py. Idempotent: drops and rebuilds the tables.
Run: python scripts/seed_db.py
"""
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from emission_factors import NON_FOOD_FACTORS

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "ecobudget.db"

# Science-based personal budget. The IPCC 1.5C pathways imply roughly 2 tonnes
# CO2e per person per year by around 2050 (and falling). We use 2,000 kg/year as
# the sustainable target, giving a monthly budget of ~166.7 kg.
ANNUAL_TARGET_KG = 2000.0
BUDGET_BASIS = ("IPCC 1.5C-aligned sustainable target of ~2 tonnes CO2e per "
                "person per year (2,000 kg/yr, ~166.7 kg/month)")


def build():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS entries;
        DROP TABLE IF EXISTS emission_factors;
        DROP TABLE IF EXISTS country_benchmarks;
        DROP TABLE IF EXISTS budget;

        CREATE TABLE emission_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            activity TEXT NOT NULL,
            unit TEXT NOT NULL,
            kg_co2e_per_unit REAL NOT NULL,
            source TEXT NOT NULL,
            year INTEGER,
            UNIQUE(category, activity)
        );
        CREATE TABLE country_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            year INTEGER,
            co2_per_capita_tonnes REAL,
            consumption_co2_per_capita_tonnes REAL
        );
        CREATE TABLE budget (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            monthly_kg_co2e REAL NOT NULL,
            annual_kg_co2e REAL NOT NULL,
            basis TEXT NOT NULL
        );
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            factor_id INTEGER REFERENCES emission_factors(id),
            category TEXT NOT NULL,
            activity TEXT NOT NULL,
            unit TEXT NOT NULL,
            quantity REAL NOT NULL,
            kg_co2e REAL NOT NULL,
            note TEXT
        );
        CREATE INDEX idx_entries_date ON entries(entry_date);
    """)

    # 1) non-food factors (DEFRA, year 2023)
    for cat, act, unit, kg, src in NON_FOOD_FACTORS:
        cur.execute(
            "INSERT INTO emission_factors (category, activity, unit, kg_co2e_per_unit, source, year)"
            " VALUES (?,?,?,?,?,?)",
            (cat, act, unit, kg, src, 2023),
        )

    # 1b) grid electricity factor (downloaded from OWID electricity carbon intensity)
    elec = json.loads((ROOT / "data" / "electricity.json").read_text())
    default_region = elec["default_region"]
    er = next(r for r in elec["regions"] if r["region"] == default_region)
    cur.execute(
        "INSERT INTO emission_factors (category, activity, unit, kg_co2e_per_unit, source, year)"
        " VALUES (?,?,?,?,?,?)",
        ("Home energy", "Electricity (grid)", "kWh", er["kg_co2e_per_kwh"],
         f"{elec['source']} [{default_region} grid, {er['gco2_per_kwh']} gCO2/kWh]", er["year"]),
    )

    # 2) food factors (downloaded from OWID / Poore & Nemecek 2018)
    food = json.loads((ROOT / "data" / "food_factors.json").read_text())
    for f in food["factors"]:
        cur.execute(
            "INSERT INTO emission_factors (category, activity, unit, kg_co2e_per_unit, source, year)"
            " VALUES (?,?,?,?,?,?)",
            ("Food", f["activity"], f["unit"], f["kg_co2e_per_unit"], food["source"], f["year"]),
        )

    # 3) country benchmarks (downloaded from OWID)
    bench = json.loads((ROOT / "data" / "benchmarks.json").read_text())
    for b in bench["benchmarks"]:
        cur.execute(
            "INSERT INTO country_benchmarks (country, year, co2_per_capita_tonnes, consumption_co2_per_capita_tonnes)"
            " VALUES (?,?,?,?)",
            (b["country"], b["year"], b["co2_per_capita_tonnes"], b.get("consumption_co2_per_capita_tonnes")),
        )

    # 4) budget
    cur.execute(
        "INSERT INTO budget (id, monthly_kg_co2e, annual_kg_co2e, basis) VALUES (1,?,?,?)",
        (round(ANNUAL_TARGET_KG / 12, 1), ANNUAL_TARGET_KG, BUDGET_BASIS),
    )
    con.commit()

    # 5) sample entries for the current month (computed from the real factors)
    def factor(activity):
        row = cur.execute(
            "SELECT id, category, unit, kg_co2e_per_unit FROM emission_factors WHERE activity=?",
            (activity,)).fetchone()
        if not row:
            raise SystemExit(f"seed sample references unknown activity: {activity}")
        return row  # (id, category, unit, kg_per_unit)

    today = date.today()
    samples = [
        # (days_ago, activity, quantity, note)
        (0, "Car - petrol (average)", 24, "Commute to office"),
        (0, "Electricity (grid)", 7.5, "Evening usage"),
        (0, "Chicken", 0.35, "Dinner"),
        (1, "National rail (train)", 60, "Trip to city"),
        (1, "Natural gas", 14, "Heating"),
        (1, "Beef", 0.25, "Lunch burger"),
        (2, "Car - petrol (average)", 24, "Commute to office"),
        (2, "Electricity (grid)", 8.2, "Working from home"),
        (2, "General waste to landfill", 1.4, "Weekly bin"),
        (3, "Flight - short-haul", 1100, "Weekend trip (one way)"),
        (3, "Coffee (per kg beans)", 0.02, "Two cups (~20 g beans)"),
        (4, "Local bus", 12, "Errands"),
        (4, "Cheese", 0.15, "Sandwich"),
        (5, "Car - electric (average)", 30, "Visiting family"),
        (5, "Vegetables (other)", 0.6, "Groceries"),
    ]
    for days_ago, activity, qty, note in samples:
        fid, cat, unit, kgpu = factor(activity)
        d = today - timedelta(days=days_ago)
        kg = round(qty * kgpu, 3)
        cur.execute(
            "INSERT INTO entries (created_ts, entry_date, factor_id, category, activity, unit, quantity, kg_co2e, note)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), d.isoformat(), fid, cat, activity, unit, qty, kg, note),
        )
    con.commit()

    nf = cur.execute("SELECT COUNT(*) FROM emission_factors").fetchone()[0]
    nb = cur.execute("SELECT COUNT(*) FROM country_benchmarks").fetchone()[0]
    ne = cur.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    total = cur.execute("SELECT ROUND(SUM(kg_co2e),1) FROM entries").fetchone()[0]
    con.close()
    print(f"Seeded {DB}")
    print(f"  emission_factors: {nf}")
    print(f"  country_benchmarks: {nb}")
    print(f"  sample entries: {ne}  (total {total} kg CO2e this month so far)")
    print(f"  monthly budget: {round(ANNUAL_TARGET_KG/12,1)} kg CO2e")


if __name__ == "__main__":
    build()
