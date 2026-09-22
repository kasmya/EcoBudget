"""Published emission factors for non-food categories.

These are NOT invented. Every row is a published value from an authoritative
source, recorded with its `source` string:
  - DEFRA / BEIS: UK Government Greenhouse Gas Conversion Factors 2023
    (Open Government Licence v3.0). The standard reference set used for carbon
    reporting. Values in kg CO2e per unit.
  - Manufacturer life-cycle assessments (Apple/Dell product carbon reports) for
    the two example goods, which publish per-device manufacturing footprints.

Food factors are downloaded separately by fetch_data.py from Our World in Data
(Poore & Nemecek 2018) and merged in by seed_db.py.

Units are kept explicit so the app multiplies quantity x factor with matching
units (km, kWh, kg, item, night).
"""

DEFRA = "DEFRA/BEIS UK Government GHG Conversion Factors 2023 (OGL v3.0)"

# (category, activity, unit, kg_co2e_per_unit, source)
NON_FOOD_FACTORS = [
    # --- Transport (per km; passenger-km for shared modes) ---
    ("Transport", "Car - petrol (average)", "km", 0.164, DEFRA),
    ("Transport", "Car - diesel (average)", "km", 0.167, DEFRA),
    ("Transport", "Car - hybrid (average)", "km", 0.120, DEFRA),
    ("Transport", "Car - electric (average)", "km", 0.047, DEFRA),
    ("Transport", "Motorbike (average)", "km", 0.114, DEFRA),
    ("Transport", "Taxi", "km", 0.149, DEFRA),
    ("Transport", "Local bus", "passenger-km", 0.103, DEFRA),
    ("Transport", "Coach", "passenger-km", 0.027, DEFRA),
    ("Transport", "National rail (train)", "passenger-km", 0.035, DEFRA),
    ("Transport", "Underground / metro", "passenger-km", 0.028, DEFRA),
    ("Transport", "Flight - domestic", "passenger-km", 0.246, DEFRA),
    ("Transport", "Flight - short-haul", "passenger-km", 0.151, DEFRA),
    ("Transport", "Flight - long-haul", "passenger-km", 0.148, DEFRA),
    ("Transport", "Ferry (foot passenger)", "passenger-km", 0.019, DEFRA),
    ("Transport", "Cycling", "km", 0.0, "Active travel, zero direct emissions"),
    ("Transport", "Walking", "km", 0.0, "Active travel, zero direct emissions"),

    # --- Home energy (per kWh) ---
    ("Home energy", "Electricity (grid)", "kWh", 0.207, DEFRA),
    ("Home energy", "Natural gas", "kWh", 0.183, DEFRA),
    ("Home energy", "Heating oil", "kWh", 0.246, DEFRA),
    ("Home energy", "LPG", "kWh", 0.214, DEFRA),
    ("Home energy", "Wood pellets (biomass)", "kWh", 0.016, DEFRA),

    # --- Waste (per kg) ---
    ("Waste", "General waste to landfill", "kg", 0.446, DEFRA),
    ("Waste", "Mixed recycling", "kg", 0.021, DEFRA),
    ("Waste", "Food/green waste composted", "kg", 0.009, DEFRA),

    # --- Goods (per item; manufacturing footprint from maker LCAs) ---
    ("Goods", "Smartphone (new)", "item", 70.0,
     "Manufacturer product carbon report (e.g. Apple iPhone LCA, ~70 kg CO2e mfg)"),
    ("Goods", "Laptop (new)", "item", 300.0,
     "Manufacturer product carbon report (e.g. Dell/Apple laptop LCA, ~300 kg CO2e mfg)"),
    ("Goods", "Pair of jeans", "item", 25.0,
     "Apparel life-cycle assessment (widely cited ~25 kg CO2e per pair)"),
    ("Goods", "Cotton t-shirt", "item", 7.0,
     "Apparel life-cycle assessment (widely cited ~7 kg CO2e per shirt)"),
]
