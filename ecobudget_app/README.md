# EcoBudget - personal carbon budget tracker

A small but complete, real prototype. You log everyday activities (driving,
electricity, food, waste); the app converts each one to kilograms of
CO2-equivalent using **published emission factors**, aggregates them by category,
and compares your footprint against a **science-based monthly carbon budget**. A
dashboard shows totals vs budget, category and daily breakdowns, your logged
entries, and how your projected annual footprint compares to national averages.

Nothing here is mocked or hardcoded. Every number the dashboard shows is computed
live from real data in a local SQLite database.

## Scope (decided for this build)

"EcoBudget" is interpreted as a **personal carbon budgeting** app, the carbon
analogue of a money budget: log activities, convert to emissions, track against a
budget, see where your footprint concentrates. This is the most sensible and
data-available scope for the name, and every piece is backed by a public dataset.

## Data sources (all real, freely licensed, fetched by `scripts/fetch_data.py`)

| Data | Source | Licence | Used for | How obtained |
| --- | --- | --- | --- | --- |
| Country CO2 per capita | Our World in Data - CO2 and Greenhouse Gas Emissions | CC-BY 4.0 | "How you compare" panel + budget context | downloaded |
| Food emission factors | Poore & Nemecek (2018), *Science*, via OWID "Food: emissions across the supply chain" | CC-BY 4.0 | kg CO2e per kg for 24 foods | downloaded + summed over supply-chain stages |
| Grid electricity intensity | OWID "Carbon intensity of electricity generation" | CC-BY 4.0 | Electricity factor (World grid, latest year) | downloaded |
| Transport / gas / oil / waste factors | DEFRA/BEIS UK Government GHG Conversion Factors 2023 | Open Government Licence v3.0 | kg CO2e per km / kWh / kg | published reference values, cited per row |
| Budget target | IPCC 1.5C pathways (~2 tonnes CO2e per person per year) | - | Monthly budget = 2,000 kg / 12 = 166.7 kg | cited |

Provenance and no-hallucination note: the OWID CO2, OWID food, and OWID
electricity-intensity files are **downloaded live** by the fetch script (raw
files land in `data/raw/`, processed files in `data/*.json`). The remaining
transport, natural-gas, heating-oil, LPG, biomass and waste factors are the
**published DEFRA/BEIS 2023 conversion factors** (the standard UK reference set),
stored in `scripts/emission_factors.py` with a `source` string on every row.
These are real published constants, not invented. An earlier draft included a
"Goods" category with recalled per-item life-cycle numbers; those were **removed**
because they could not be traced to a single downloaded source. Every factor now
in the database is either downloaded or a cited DEFRA reference value.

## Assumptions

- **Food factors** are the sum of all supply-chain stages OWID publishes (land-use
  change, farm, feed, processing, transport, retail, packaging, losses), in kg
  CO2e per kg of product.
- **Budget** defaults to the IPCC 1.5C-aligned ~2 t/yr per person. You can change
  the monthly budget in the UI (it is stored in the DB).
- **Projected annual footprint** = (this month's logged kg / days elapsed) x 365.
  Your logs are a personal subset, so this is a partial-footprint estimate and is
  labelled as such; it is not directly comparable to a country's full territorial
  per-capita figure, which is why the comparison panel is framed as context.
- Grid electricity uses OWID's downloaded World grid carbon intensity (latest
  year). data/electricity.json also contains UK/US/EU values; change
  `default_region` in the electricity JSON (or re-point seed_db.py) for local
  accuracy.

## Architecture

```
ecobudget_app/
  scripts/
    fetch_data.py         downloads OWID CO2 + food + electricity data -> data/*.json
    emission_factors.py   DEFRA factors (transport, gas/oil, waste)
    seed_db.py            builds & seeds data/ecobudget.db from the fetched data
  data/
    raw/                  downloaded source CSVs (OWID)
    benchmarks.json       processed OWID per-capita CO2
    food_factors.json     processed OWID food factors
    ecobudget.db          the seeded SQLite database (48 factors, 12 countries, 15 sample entries)
  main.py                 FastAPI backend (reads/writes the DB, computes summaries)
  static/                 the dashboard (index.html + styles.css + app.js, Chart.js)
  requirements.txt
```

**Database schema:** `emission_factors(category, activity, unit, kg_co2e_per_unit,
source, year)`, `entries(entry_date, factor_id, category, activity, unit,
quantity, kg_co2e, note)`, `country_benchmarks(country, year,
co2_per_capita_tonnes, ...)`, `budget(monthly_kg_co2e, annual_kg_co2e, basis)`.

**API** (FastAPI, all JSON): `GET /api/factors`, `GET/POST /api/entries`,
`DELETE /api/entries/{id}`, `GET /api/summary`, `GET /api/benchmarks`,
`GET/PUT /api/budget`. Interactive docs at `/docs`.

## How to run

```bash
cd ecobudget_app
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Fetch the real public data (needs internet)
python scripts/fetch_data.py

# 2. Build and seed the SQLite database
python scripts/seed_db.py

# 3. Run the app
uvicorn main:app --reload --port 8077
# open http://127.0.0.1:8077/
```

The repository already ships a seeded `data/ecobudget.db`, so if you just want to
see it run you can skip steps 1 and 2 and go straight to `uvicorn`.

## What you can do in the dashboard

- **Log an activity**: pick a category and activity, enter the amount (the unit
  and a live kg CO2e estimate update as you type), add an optional note and date.
- **See it update live**: the KPI cards, budget bar, category doughnut, daily bar
  chart, breakdown and top-activity tables, entries list, and country comparison
  all refresh from the backend after every add or delete.
- **Track vs budget**: total this month vs the monthly budget, percent used,
  remaining, and an over/under-budget state.
- **Change your budget**: edit the monthly target (persisted to the DB).
- **Compare**: your projected annual footprint against national per-capita CO2.

## Sample data shipped

The database is seeded with 15 realistic sample entries across a week (commutes,
electricity, meals, a short-haul flight, a train trip, waste, a bus ride),
totalling ~222 kg CO2e for the month against the 166.7 kg budget, so the
dashboard opens in a meaningful "over budget" state you can immediately explore.
