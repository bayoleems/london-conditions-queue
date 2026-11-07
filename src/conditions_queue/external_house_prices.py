"""External feature example: local house prices per square metre, by borough and month.

Source: hpm_la_2024.zip (House Price per Square Metre, Price Paid x EPC linked; GLA / UCL via
London Datastore - see Data Asset Register row 36). One CSV per local authority, transactions to 2024.

Writes outputs/ext_house_prices.csv with one row per (area_name, month):
  ppm2_12m      median price per m2 over the 12 months up to and including that month
  ppm2_yoy      % change in ppm2_12m against 12 months earlier
Models must join this AS OF the month BEFORE submission, so no future prices leak in.

Run: python -m conditions_queue.external_house_prices   (needs data/raw/hpm_la_2024.zip)
"""
import zipfile

import pandas as pd

from conditions_queue.paths import HOUSE_PRICES_CSV as OUT
from conditions_queue.paths import HOUSE_PRICES_ZIP as ZIP

# PlanIt area_name -> local authority file(s). The two development corporations span several boroughs.
AREA_FILES = {
    "Kensington": ["Kensington_and_Chelsea"], "Kingston": ["Kingston_upon_Thames"],
    "Richmond": ["Richmond_upon_Thames"], "City": ["City_of_London"],
    "Old Oak Park Royal": ["Ealing", "Hammersmith_and_Fulham", "Brent"],
    "London Legacy": ["Newham", "Tower_Hamlets", "Hackney", "Waltham_Forest"],
}
BOROUGHS = ["Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "Croydon", "Ealing", "Enfield",
            "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
            "Hounslow", "Islington", "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", "Southwark", "Sutton",
            "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"]
for b in BOROUGHS:
    AREA_FILES.setdefault(b, [b.replace(" ", "_")])


def load_la(z, la):
    name = f"hpm_la_2024/{la}_link_26122024.csv"
    with z.open(name) as f:
        t = pd.read_csv(f, usecols=["priceper", "dateoftransfer"], parse_dates=["dateoftransfer"])
    t = t[(t["dateoftransfer"] >= "2017-01-01") & t["priceper"].between(500, 100_000)]
    return t


def build():
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        cache = {}
        for area, las in AREA_FILES.items():
            t = pd.concat([cache.setdefault(la, load_la(z, la)) for la in las])
            t["month"] = t["dateoftransfer"].dt.to_period("M")
            months = pd.period_range("2018-01", t["month"].max(), freq="M")
            for m in months:
                w = t[(t["month"] > m - 12) & (t["month"] <= m)]["priceper"]
                rows.append({"area_name": area, "month": m.to_timestamp(), "ppm2_12m": w.median(), "n_sales_12m": len(w)})
    ext = pd.DataFrame(rows).sort_values(["area_name", "month"])
    ext["ppm2_yoy"] = ext.groupby("area_name")["ppm2_12m"].pct_change(12) * 100
    ext.to_csv(OUT, index=False)
    return ext


if __name__ == "__main__":
    e = build()
    print(f"wrote {OUT}: {len(e):,} rows, {e['area_name'].nunique()} areas, {e['month'].min():%Y-%m} to {e['month'].max():%Y-%m}")
    print(e[e["month"] == e["month"].max()].sort_values("ppm2_12m").round(1).to_string(index=False))
