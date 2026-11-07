"""Every file location the project uses, in one place.

Raw and external data live under data/ and are not committed (see data/README.md).
Derived tables live under outputs/. The GitHub Pages site lives under site/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PLANNING_DATA = DATA / "external" / "planning_data"
PARQUET = next(RAW.glob("foundations_london_housing_*.parquet"), RAW / "foundations_london_housing.parquet")
HOUSE_PRICES_ZIP = RAW / "hpm_la_2024.zip"

OUTPUTS = ROOT / "outputs"
SUMMARY_JSON = OUTPUTS / "summary.json"
DISCHARGE_CSV = OUTPUTS / "discharge_apps.csv"
SCHEMES_CSV = OUTPUTS / "schemes.csv"
JEV_JSONL = OUTPUTS / "jev_topics.jsonl"
HOUSE_PRICES_CSV = OUTPUTS / "ext_house_prices.csv"
CONSTRAINTS_CSV = OUTPUTS / "ext_constraints.csv"
UI_LATE_JSON = OUTPUTS / "ui_late_model.json"
UI_DAYS_JSON = OUTPUTS / "ui_days_model.json"
UI_DASHBOARD_JSON = OUTPUTS / "ui_dashboard.json"

TEMPLATES = Path(__file__).resolve().parent / "templates"
SITE = ROOT / "site"
SITE_DATA_JSON = SITE / "data" / "ui_dashboard.json"
SITE_INSIGHT_HTML = SITE / "insight.html"

ENV_FILE = ROOT / ".env"
