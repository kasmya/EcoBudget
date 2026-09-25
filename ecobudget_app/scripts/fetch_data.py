"""Fetch REAL public data for EcoBudget and write processed JSON.

Primary source (downloaded live):
  Our World in Data - CO2 and Greenhouse Gas Emissions dataset (CC-BY 4.0).
  https://github.com/owid/co2-data  (mirror on owid-public spaces).
  We use per-capita CO2 by country/year for the country-comparison panel and to
  sanity-check the science-based personal budget.

Emission factors (kg CO2e per unit of activity) are NOT invented: they are
published values from DEFRA (UK Government GHG Conversion Factors 2023, Open
Government Licence), the US EPA, and IPCC/Our World in Data. They are defined in
scripts/emission_factors.py with a `source` string on every row and are seeded
into the database by seed_db.py. This script only handles the OWID download.

Run: python scripts/fetch_data.py
"""
import csv
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

OWID_URLS = [
    "https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv",
    "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
]
# OWID "Food: greenhouse gas emissions across the supply chain" (Poore & Nemecek
# 2018, Science; CC-BY). Per-kg CO2e broken into supply-chain stages.
FOOD_URL = "https://ourworldindata.org/grapher/food-emissions-supply-chain.csv"
# OWID "Carbon intensity of electricity generation" (gCO2/kWh), CC-BY. Used to
# ground the electricity emission factor in downloaded data rather than a
# recalled constant.
ELEC_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"
ELEC_REGIONS = ["World", "United Kingdom", "United States", "European Union (27)"]

# Human-friendly logging labels + serving-size hints for common foods.
FOOD_LABELS = {
    "Beef (beef herd)": "Beef", "Beef (dairy herd)": "Beef (from dairy cows)",
    "Lamb & Mutton": "Lamb", "Cheese": "Cheese", "Pig Meat": "Pork",
    "Poultry Meat": "Chicken", "Eggs": "Eggs", "Fish (farmed)": "Fish (farmed)",
    "Prawns (farmed)": "Prawns", "Milk": "Milk", "Tofu": "Tofu", "Rice": "Rice",
    "Wheat & Rye": "Bread / wheat", "Maize": "Maize", "Potatoes": "Potatoes",
    "Tomatoes": "Tomatoes", "Bananas": "Bananas", "Apples": "Apples",
    "Nuts": "Nuts", "Peas": "Peas", "Coffee": "Coffee (per kg beans)",
    "Dark Chocolate": "Dark chocolate", "Cane Sugar": "Sugar",
    "Other Vegetables": "Vegetables (other)", "Root Vegetables": "Root vegetables",
}
STAGE_COLS = ["Land use change", "Farm", "Animal feed", "Processing",
              "Transport", "Retail", "Packaging", "Losses"]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# Countries to surface in the comparison panel (plus World average).
BENCHMARK_COUNTRIES = [
    "World", "United States", "Australia", "Canada", "Germany", "United Kingdom",
    "China", "Japan", "France", "India", "Brazil", "Kenya",
]


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    raw = None
    for url in OWID_URLS:
        try:
            print(f"Downloading OWID CO2 data from {url} ...")
            raw = download(url)
            (RAW / "owid-co2-data.csv").write_bytes(raw)
            print(f"  saved {len(raw):,} bytes to data/raw/owid-co2-data.csv")
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
    if raw is None:
        raise SystemExit("Could not download OWID data (no network?). Aborting.")

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    # keep latest non-empty per-capita value per country of interest
    latest = {}  # country -> (year, co2_per_capita_t, consumption_per_capita_t|None)
    wanted = set(BENCHMARK_COUNTRIES)
    for row in reader:
        c = row.get("country")
        if c not in wanted:
            continue
        try:
            year = int(row["year"])
        except (ValueError, KeyError):
            continue
        pc = row.get("co2_per_capita") or ""
        cons = row.get("consumption_co2_per_capita") or ""
        if pc == "":
            continue
        prev = latest.get(c)
        if prev is None or year > prev[0]:
            latest[c] = (year, float(pc), float(cons) if cons != "" else None)

    benchmarks = []
    for c in BENCHMARK_COUNTRIES:
        if c in latest:
            yr, pc, cons = latest[c]
            benchmarks.append({
                "country": c,
                "year": yr,
                "co2_per_capita_tonnes": round(pc, 3),
                "consumption_co2_per_capita_tonnes": round(cons, 3) if cons is not None else None,
            })
    benchmarks.sort(key=lambda b: b["co2_per_capita_tonnes"], reverse=True)

    out = {
        "source": "Our World in Data - CO2 and Greenhouse Gas Emissions (CC-BY 4.0), "
                  "https://github.com/owid/co2-data",
        "metric": "Annual CO2 emissions per capita (tonnes). Production-based; "
                  "consumption-based shown where available.",
        "benchmarks": benchmarks,
    }
    (ROOT / "data" / "benchmarks.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/benchmarks.json with {len(benchmarks)} countries "
          f"(latest years {min(b['year'] for b in benchmarks)}-{max(b['year'] for b in benchmarks)}).")
    for b in benchmarks:
        print(f"  {b['country']:<16} {b['co2_per_capita_tonnes']:>6.2f} t/yr ({b['year']})")

    # ---- Food emission factors (downloaded, Poore & Nemecek 2018 via OWID) ----
    print(f"\nDownloading OWID food supply-chain emissions from {FOOD_URL} ...")
    fraw = download(FOOD_URL)
    (RAW / "food-emissions-supply-chain.csv").write_bytes(fraw)
    freader = csv.DictReader(io.StringIO(fraw.decode("utf-8")))
    foods = []
    for row in freader:
        ent = row.get("Entity")
        if ent not in FOOD_LABELS:
            continue
        total = 0.0
        for col in STAGE_COLS:
            v = row.get(col, "") or "0"
            try:
                total += float(v)
            except ValueError:
                pass
        foods.append({
            "activity": FOOD_LABELS[ent],
            "owid_entity": ent,
            "unit": "kg",
            "kg_co2e_per_unit": round(total, 3),
            "year": int(row.get("Year", 2018) or 2018),
        })
    foods.sort(key=lambda f: f["kg_co2e_per_unit"], reverse=True)
    food_out = {
        "source": "Poore, J. & Nemecek, T. (2018), Science, via Our World in Data "
                  "'Food: greenhouse gas emissions across the supply chain' (CC-BY 4.0)",
        "metric": "kg CO2e per kg of food product, summed across supply-chain stages",
        "factors": foods,
    }
    (ROOT / "data" / "food_factors.json").write_text(json.dumps(food_out, indent=2))
    print(f"Wrote data/food_factors.json with {len(foods)} foods. Sample:")
    for f in foods[:6]:
        print(f"  {f['activity']:<20} {f['kg_co2e_per_unit']:>7.2f} kg CO2e/kg")

    # ---- Electricity carbon intensity (downloaded, OWID) ----
    print(f"\nDownloading OWID electricity carbon intensity from {ELEC_URL} ...")
    eraw = download(ELEC_URL)
    (RAW / "carbon-intensity-electricity.csv").write_bytes(eraw)
    ereader = csv.DictReader(io.StringIO(eraw.decode("utf-8")))
    icol = "Carbon intensity"
    elat = {}  # region -> (year, gco2_per_kwh)
    for row in ereader:
        e = row.get("Entity")
        if e not in ELEC_REGIONS or not row.get(icol):
            continue
        try:
            y = int(row["Year"]); v = float(row[icol])
        except (ValueError, KeyError):
            continue
        if e not in elat or y > elat[e][0]:
            elat[e] = (y, v)
    elec_out = {
        "source": "Our World in Data - Carbon intensity of electricity generation "
                  "(CC-BY 4.0), https://ourworldindata.org/grapher/carbon-intensity-electricity",
        "metric": "gCO2 per kWh; kg_co2e_per_kwh = value / 1000",
        "default_region": "World",
        "regions": [
            {"region": r, "year": elat[r][0],
             "gco2_per_kwh": round(elat[r][1], 1),
             "kg_co2e_per_kwh": round(elat[r][1] / 1000, 4)}
            for r in ELEC_REGIONS if r in elat
        ],
    }
    (ROOT / "data" / "electricity.json").write_text(json.dumps(elec_out, indent=2))
    print("Wrote data/electricity.json (grid carbon intensity):")
    for r in elec_out["regions"]:
        print(f"  {r['region']:<22} {r['kg_co2e_per_kwh']:.4f} kg/kWh ({r['year']})")


if __name__ == "__main__":
    main()
