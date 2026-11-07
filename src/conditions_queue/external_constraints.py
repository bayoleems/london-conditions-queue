"""External feature example 2: site constraints from planning.data.gov.uk, joined on location.

Inputs (national bulk CSVs from https://files.planning.data.gov.uk/dataset/<name>.csv, fetched Nov 2026,
each in its own folder under data/external/planning_data/): conservation-area, article-4-direction-area,
tree-preservation-zone, listed-building, brownfield-land. Flood risk zones were skipped (2.5 GB).

For every discharge application with a location, writes outputs/ext_constraints.csv:
  in_conservation_area, in_article4_area, in_tpo_zone   site point falls inside the designated area
  listed_within_100m                                    listed buildings within 100 m
  brownfield_within_50m                                 a brownfield register site within 50 m
Only designations in force on the application's start date count (start-date <= submission, no earlier end-date),
so nothing designated later leaks in.

Run: python -m conditions_queue.external_constraints
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from conditions_queue.paths import CONSTRAINTS_CSV as OUT
from conditions_queue.paths import DISCHARGE_CSV as APPS
from conditions_queue.paths import PARQUET
from conditions_queue.paths import PLANNING_DATA as SRC
BBOX = (-0.56, 51.25, 0.38, 51.73)  # London, with a margin
BNG = "EPSG:27700"                   # British National Grid, metres


def load_layer(name, use_point=False):
    df = pd.read_csv(SRC / name / f"{name}.csv", dtype=str,
                     usecols=lambda c: c in {"entity", "geometry", "point", "start-date", "end-date", "organisation-entity"})
    col = "point" if use_point else "geometry"
    df[col] = df[col].fillna(df["point"] if not use_point else df["geometry"])
    df = df[df[col].notna()].copy()
    # quick London pre-filter on the representative point before parsing every shape
    pts = shapely.from_wkt(df["point"].fillna("POINT EMPTY"), on_invalid="ignore")
    x, y = shapely.get_x(pts), shapely.get_y(pts)
    near = np.isnan(x) | ((x > BBOX[0]) & (x < BBOX[2]) & (y > BBOX[1]) & (y < BBOX[3]))
    df = df[near].copy()
    geom = shapely.from_wkt(df[col], on_invalid="ignore")
    g = gpd.GeoDataFrame(df.drop(columns=["geometry", "point"]), geometry=geom, crs="EPSG:4326")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]
    g["start"] = pd.to_datetime(g["start-date"], errors="coerce")
    g["end"] = pd.to_datetime(g["end-date"], errors="coerce")
    g = g.to_crs(BNG)
    g["geometry"] = g.geometry.buffer(0) if not use_point else g.geometry  # repair invalid polygons
    print(f"  {name}: {len(g):,} features in London")
    return g[["entity", "start", "end", "geometry"]]


def in_force(j, when):
    """Keep matches whose designation had started (or has no date) and not ended by the submission date."""
    return (j["start"].isna() | (j["start"] <= when)) & (j["end"].isna() | (j["end"] > when))


def main():
    apps = pd.read_csv(APPS, usecols=["name", "start_date"], parse_dates=["start_date"])
    loc = pd.read_parquet(PARQUET, columns=["name", "lat", "lng"])
    apps = apps.merge(loc, on="name").dropna(subset=["lat", "lng"])
    pts = gpd.GeoDataFrame(apps, geometry=gpd.points_from_xy(apps["lng"], apps["lat"]), crs="EPSG:4326").to_crs(BNG)
    print(f"{len(pts):,} applications with a location")
    out = pts[["name"]].copy()

    print("Loading layers")
    for name, col in [("conservation-area", "in_conservation_area"),
                      ("article-4-direction-area", "in_article4_area"),
                      ("tree-preservation-zone", "in_tpo_zone")]:
        layer = load_layer(name)
        j = gpd.sjoin(pts, layer, how="inner", predicate="within")
        j = j[in_force(j, j["start_date"])]
        out[col] = out.index.isin(j.index).astype(int)

    for name, col, metres in [("listed-building", "listed_within_100m", 100),
                              ("brownfield-land", "brownfield_within_50m", 50)]:
        layer = load_layer(name, use_point=True)
        layer["geometry"] = layer.geometry.centroid if name == "brownfield-land" else layer.geometry
        j = gpd.sjoin(pts.assign(geometry=pts.buffer(metres)), layer, how="inner", predicate="intersects")
        j = j[in_force(j, j["start_date"])]
        counts = j.groupby(level=0).size()
        out[col] = counts.reindex(out.index).fillna(0).astype(int)
        if name == "brownfield-land":
            out[col] = (out[col] > 0).astype(int)

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(out.drop(columns="name").mean().round(3).to_string())


if __name__ == "__main__":
    main()
