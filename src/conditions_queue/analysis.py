"""How long does it take London schemes to clear planning conditions after permission is granted?

Picks the discharge-of-condition applications out of the raw extract, tags their topics and condition
counts with keywords, matches them to their original permission, and summarises timeliness.

Input : data/raw/foundations_london_housing_*.parquet (PlanIt-derived, London)
Output: outputs/summary.json, outputs/discharge_apps.csv, outputs/schemes.csv

Run   : python -m conditions_queue.analysis
"""
import collections
import csv
import datetime as dt
import json
import re
import statistics as st

import duckdb

from conditions_queue.paths import DISCHARGE_CSV, OUTPUTS, PARQUET, SCHEMES_CSV, SUMMARY_JSON

SRC = str(PARQUET)
OUTPUTS.mkdir(exist_ok=True)

STAT_DAYS = 56          # 8-week statutory period for approval of details (DMPO 2015 art. 27)
MIN_BOROUGH_N = 300     # boroughs with fewer discharge apps are too thinly scraped to compare
DATA_END = dt.date(2025, 12, 31)   # last submission date in the extract

# ---------------------------------------------------------------- classifiers
DISCHARGE = re.compile(
    r"details (pursuant|of|submitted|required|in respect)|discharge|approval of details|"
    r"submission of details|pursuant to condition|required by condition", re.I)
NOT_DISCHARGE = re.compile(
    r"variation|vary|removal of condition|remove condition|section 73|\bs\.?73|"
    r"minor material|non[- ]material|reserved matters", re.I)
RESIDENTIAL = re.compile(
    r"dwelling|\bflats?\b|residential|\bhomes?\b|\bhouses?\b|\bunits?\b|\bC3\b|apartment|"
    r"maisonette|\bHMO\b", re.I)

TOPICS = {
    "Construction logistics / management": r"construction (logistics|management|method)|\bCLP\b|\bCMP\b|\bCEMP\b|demolition (management|method)",
    "Materials & design detail": r"material|sample|facing|brick|cladding|elevation|architectural|window|reveal|detailed design|balustrad",
    "Landscaping & trees": r"landscap|\btrees?\b|arboric|planting|tree protection",
    "Drainage & flood": r"drainage|\bsuds\b|sustainable (urban )?drainage|flood|surface water",
    "Contamination & ground": r"contaminat|remediation|ground investigation|site investigation|land quality|verification report",
    "Energy & carbon": r"energy|carbon|\bco2\b|breeam|sustainab|overheating|photovoltaic|\bpv\b|heat pump|water (usage|consumption|efficiency)",
    "Noise, vibration & air": r"noise|acoustic|vibration|sound insulation|air quality|ventilation|odour",
    "Ecology & biodiversity": r"ecolog|biodiversity|\bbats?\b|\bbirds?\b|green roof|living roof|invasive|habitat",
    "Archaeology & heritage": r"archaeolog|written scheme of investigation|\bWSI\b|heritage|recording",
    "Waste, cycle & parking": r"refuse|waste|recycling|cycle|parking|electric vehicle|\bEV\b|car club",
    "Fire safety": r"\bfire\b",
    "Accessibility": r"M4\(?[23]|wheelchair|accessib",
    "Piling & structure": r"piling|foundation|basement|structural|ground floor slab",
    "Highways & travel": r"highway|s278|section 278|travel plan|delivery and servicing|servicing plan",
    "Boundary & lighting": r"boundary|lighting|external lights",
    "Secured by Design": r"secured by design|\bSBD\b",
}
TOPICS = {k: re.compile(v, re.I) for k, v in TOPICS.items()}
# Topics that are usually attached as pre-commencement conditions (work cannot start until cleared)
PRECOMMENCEMENT = {"Construction logistics / management", "Contamination & ground", "Archaeology & heritage",
                   "Drainage & flood", "Piling & structure", "Landscaping & trees"}

DWELL = re.compile(
    r"(\d{1,4})\s*(?:x\s*)?(?:no\.?\s*)?(?:new\s+)?(?:self[- ]contained\s+)?(?:residential\s+)?"
    r"(?:units|dwellings?|flats?|homes|houses|apartments|maisonettes)\b", re.I)
REF_TOKEN = re.compile(r"[A-Z0-9][A-Z0-9/.\-]*\d[A-Z0-9/.\-]*", re.I)
DATED = re.compile(r"dated:?\s*(\d{1,2})(?:st|nd|rd|th)?[./\- ]+(\d{1,2}|[A-Za-z]{3,9})[./\- ]+(\d{2,4})", re.I)
COND_LIST = re.compile(r"conditions?\s+((?:\d{1,3}[a-z]?(?:\s*\([^)]{0,80}\))?\s*(?:,|and|&|to|-)?\s*)+)", re.I)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def norm_ref(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def strip_suffix(n):
    m = re.match(r"^(.*?\d)[A-Z]*$", n)
    if m and m.group(1) != n and sum(ch.isdigit() for ch in m.group(1)) >= 5:
        return m.group(1)
    return None


def outcome(decision, status):
    d = (decision or "").lower()
    s = (status or "").lower()
    if "withdraw" in d or s == "withdrawn":
        return "withdrawn"
    if "part" in d or "split" in d:
        return "partial"
    if re.search(r"refus|not approved|reject", d) or (s == "rejected" and not d):
        return "refused"
    if s == "rejected":
        return "refused"
    if s in ("permitted", "conditions") or re.search(r"grant|approv|agreed|discharg", d):
        return "approved"
    if s == "undecided" or not d:
        return "pending"
    return "other"


def parse_dated(text):
    m = DATED.search(text)
    if not m:
        return None
    d, mo, y = m.groups()
    try:
        mo = int(mo) if mo.isdigit() else MONTHS.get(mo[:3].lower())
        y = int(y)
        y = y + 2000 if y < 100 else y
        return dt.date(y, mo, int(d))
    except (TypeError, ValueError):
        return None


def n_conditions(text):
    m = COND_LIST.search(text)
    if not m:
        return 1
    nums = re.findall(r"(?<![(\w])(\d{1,3})[a-z]?(?![^(]*\))", m.group(1))
    return max(1, len(set(nums)))


def dwellings(text, fallback):
    vals = [int(x) for x in DWELL.findall(text) if 0 < int(x) < 5000]
    if vals:
        return max(vals)
    return int(fallback) if fallback and fallback < 5000 else None


def band(n):
    if n is None:
        return "Unknown"
    if n < 10:
        return "Minor (1-9 homes)"
    if n < 150:
        return "Major (10-149)"
    return "Strategic (150+)"


def med(xs):
    return round(st.median(xs), 1) if xs else None


# ---------------------------------------------------------------- load
con = duckdb.connect()
cols = ["uid", "area_name", "name", "app_type", "app_size", "decision", "status",
        "start_date", "decided_date", "days_to_decision", "description", "n_dwellings"]
rows = [dict(zip(cols, r)) for r in con.sql(f"select {', '.join(cols)} from '{SRC}'").fetchall()]

# index every application by normalised reference, per borough, for parent lookup
idx = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    r["ref"] = r["name"].split("/", 1)[1] if "/" in r["name"] else r["name"]
    n = norm_ref(r["ref"])
    idx[r["area_name"]][n].append(r)
    s = strip_suffix(n)
    if s:
        idx[r["area_name"]][s].append(r)


def find_parent(r):
    own = norm_ref(r["ref"])
    for t in REF_TOKEN.findall(r["description"]):
        n = norm_ref(t)
        if sum(ch.isdigit() for ch in n) < 4 or n == own:
            continue
        for key in (n, strip_suffix(n)):
            if not key:
                continue
            c = [x for x in idx[r["area_name"]].get(key, [])
                 if x["uid"] != r["uid"] and not DISCHARGE.search(x["description"] or "")
                 and x["decided_date"] and r["start_date"] and x["decided_date"] <= r["start_date"]]
            if c:
                return c[0]
    return None


# ---------------------------------------------------------------- discharge apps
dis = []
for r in rows:
    t = r["description"] or ""
    if not (DISCHARGE.search(t) and not NOT_DISCHARGE.search(t) and RESIDENTIAL.search(t)):
        continue
    r["outcome"] = outcome(r["decision"], r["status"])
    r["topics"] = [k for k, rx in TOPICS.items() if rx.search(t)]
    r["n_cond"] = n_conditions(t)
    parent = find_parent(r)
    r["parent"] = parent
    r["parent_uid"] = parent["name"] if parent else None
    r["parent_date"] = parent["decided_date"] if parent else parse_dated(t)
    r["homes"] = dwellings(t, r["n_dwellings"] or (parent and parent["n_dwellings"]))
    if r["homes"] is None and parent:
        r["homes"] = dwellings(parent["description"] or "", None)
    r["band"] = band(r["homes"])
    r["year"] = r["start_date"].year if r["start_date"] else None
    d = r["days_to_decision"]
    if r["outcome"] == "pending" and r["start_date"]:
        r["age"] = (DATA_END - r["start_date"]).days
    r["over8"] = (d is not None and d > STAT_DAYS) or (r["outcome"] == "pending" and r.get("age", 0) > STAT_DAYS)
    r["lag"] = ((r["start_date"] - r["parent_date"]).days
                if r["parent_date"] and r["start_date"] and r["parent_date"] <= r["start_date"] else None)
    dis.append(r)

decided = [r for r in dis if r["outcome"] in ("approved", "refused", "partial") and r["days_to_decision"] is not None
           and r["days_to_decision"] >= 0]
eligible = [r for r in dis if r["outcome"] != "withdrawn" and r["year"] and r["year"] >= 2021]


def block(rs, ds):
    ds = [r["days_to_decision"] for r in ds]
    resolved = [r for r in rs if r["outcome"] in ("approved", "refused", "partial")]
    return {
        "n": len(rs),
        "median_days": med(ds),
        "p75_days": round(st.quantiles(ds, n=4)[2], 1) if len(ds) > 3 else None,
        "pct_over_8wk": round(100 * sum(r["over8"] for r in rs) / len(rs), 1) if rs else None,
        "pct_not_approved": round(100 * sum(r["outcome"] in ("refused", "partial") for r in resolved) / len(resolved), 1) if resolved else None,
    }


summary = {"source": PARQUET.name, "statutory_days": STAT_DAYS}
summary["overall"] = block(eligible, [r for r in decided if r["year"] and r["year"] >= 2021])
summary["overall"]["n_all_discharge"] = len(dis)
summary["overall"]["n_matched_parent"] = sum(1 for r in dis if r["parent_uid"])
summary["overall"]["n_with_parent_date"] = sum(1 for r in dis if r["lag"] is not None)
summary["overall"]["outcomes"] = dict(collections.Counter(r["outcome"] for r in dis))

dec21 = [r for r in decided if r["year"] and r["year"] >= 2021]


def by(key, rs_all, rs_dec, min_n=1):
    groups = collections.defaultdict(list)
    for r in rs_all:
        for k in (key(r) if isinstance(key(r), list) else [key(r)]):
            groups[k].append(r)
    dgroups = collections.defaultdict(list)
    for r in rs_dec:
        for k in (key(r) if isinstance(key(r), list) else [key(r)]):
            dgroups[k].append(r)
    return {k: block(v, dgroups[k]) for k, v in groups.items() if len(v) >= min_n and k is not None}


summary["by_year"] = by(lambda r: r["year"], eligible, dec21)
summary["by_band"] = by(lambda r: r["band"], eligible, dec21)
summary["by_topic"] = by(lambda r: r["topics"], eligible, dec21, 200)
single = [r for r in eligible if len(r["topics"]) == 1]
summary["by_topic_single"] = by(lambda r: r["topics"][0], single, [r for r in dec21 if len(r["topics"]) == 1], 100)
summary["by_ncond"] = by(lambda r: "1" if r["n_cond"] == 1 else "2-3" if r["n_cond"] <= 3 else "4-6" if r["n_cond"] <= 6 else "7+",
                         eligible, dec21)
boro_n = collections.Counter(r["area_name"] for r in eligible)
summary["by_borough"] = by(lambda r: r["area_name"] if boro_n[r["area_name"]] >= MIN_BOROUGH_N else None,
                           eligible, dec21)
summary["boroughs_excluded_thin"] = {k: v for k, v in boro_n.items() if v < MIN_BOROUGH_N}

# deadline behaviour: decided apps only (no pending), weekly bins ending on day 7k
dd = [r["days_to_decision"] for r in dec21]
summary["decided_only_pct_over_8wk"] = round(100 * sum(d > STAT_DAYS for d in dd) / len(dd), 1)
summary["pct_decided_in_final_week"] = round(100 * sum(50 <= d <= 56 for d in dd) / len(dd), 1)
summary["pct_over_26wk"] = round(100 * sum(d > 182 for d in dd) / len(dd), 1)
hist = collections.Counter(min(-(-d // 7), 27) for d in dd)
summary["weekly_histogram"] = {"note": "week k = days 7k-6..7k; 27 = over 26 weeks", "bins": [hist.get(k, 0) for k in range(28)]}

# lag from permission to first discharge submission (applicant side)
lags = [r["lag"] for r in dis if r["lag"] is not None and r["lag"] <= 3650]
summary["lag_permission_to_submission"] = {"n": len(lags), "median_days": med(lags),
                                           "p25": st.quantiles(lags, n=4)[0], "p75": st.quantiles(lags, n=4)[2]}

# ---------------------------------------------------------------- per-scheme view (matched parents)
schemes = collections.defaultdict(list)
for r in dis:
    if r["parent_uid"]:
        schemes[r["parent_uid"]].append(r)
scheme_rows = []
for pu, apps in schemes.items():
    p = apps[0]["parent"]
    apps.sort(key=lambda r: r["start_date"])
    ends = [r["decided_date"] for r in apps if r["decided_date"]]
    resolved = [r for r in apps if r["outcome"] in ("approved", "refused", "partial")]
    homes = max([r["homes"] for r in apps if r["homes"]] or [0]) or None
    scheme_rows.append({
        "parent": p["name"], "borough": p["area_name"], "granted": p["decided_date"].isoformat(),
        "homes": homes, "band": band(homes), "n_discharge_apps": len(apps),
        "n_conditions_listed": sum(r["n_cond"] for r in apps),
        "n_refused": sum(r["outcome"] in ("refused", "partial") for r in resolved),
        "days_to_first_submission": (apps[0]["start_date"] - p["decided_date"]).days,
        "days_grant_to_last_decision": (max(ends) - p["decided_date"]).days if ends else None,
        "council_days_total": sum(r["days_to_decision"] or 0 for r in resolved),
    })
# limit to permissions granted 2021-2023 so every scheme had >= 2 years to come back
mature = [s for s in scheme_rows if "2021" <= s["granted"][:4] <= "2023"]
sb = collections.defaultdict(list)
for s in mature:
    sb[s["band"]].append(s)
sb["All"] = mature
summary["schemes_granted_2021_23"] = {
    b: {"n_schemes": len(v),
        "median_discharge_apps": med([s["n_discharge_apps"] for s in v]),
        "mean_discharge_apps": round(sum(s["n_discharge_apps"] for s in v) / len(v), 2),
        "pct_with_refusal": round(100 * sum(s["n_refused"] > 0 for s in v) / len(v), 1),
        "median_days_to_first_submission": med([s["days_to_first_submission"] for s in v]),
        "median_days_grant_to_last_decision": med([s["days_grant_to_last_decision"] for s in v if s["days_grant_to_last_decision"] is not None]),
        "homes": sum(s["homes"] or 0 for s in v)}
    for b, v in sb.items() if len(v) >= 20}

# homes caught in a refused discharge (each refusal = resubmission round)
ref_rounds = [r for r in dis if r["outcome"] in ("refused", "partial")]
summary["refusal_rounds"] = {"n": len(ref_rounds), "homes_in_schemes_affected": None}
resubs = []
for pu, apps in schemes.items():
    for i, a in enumerate(apps):
        if a["outcome"] in ("refused", "partial") and a["decided_date"]:
            nxt = [b for b in apps[i + 1:] if set(b["topics"]) & set(a["topics"]) and b["start_date"] >= a["start_date"]]
            if nxt and nxt[0]["decided_date"] and nxt[0]["outcome"] == "approved":
                resubs.append((nxt[0]["decided_date"] - a["start_date"]).days - (a["days_to_decision"] or 0))
summary["refusal_rounds"]["n_resubmission_pairs"] = len(resubs)
summary["refusal_rounds"]["median_extra_days"] = med(resubs)

# ---------------------------------------------------------------- write
def cell(value):
    return "" if value is None else "|".join(value) if isinstance(value, list) else value


def write_csv(path, keys, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(keys) + "\n")
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for row in rows:
            w.writerow([cell(row[k]) for k in keys])


with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2, default=str)
write_csv(DISCHARGE_CSV, ["name", "area_name", "start_date", "decided_date", "days_to_decision", "outcome", "over8",
                          "band", "homes", "n_cond", "topics", "parent_uid", "parent_date", "lag"], dis)
write_csv(SCHEMES_CSV, list(scheme_rows[0].keys()), scheme_rows)
print(f"wrote {SUMMARY_JSON.name}, {DISCHARGE_CSV.name} ({len(dis):,} rows), {SCHEMES_CSV.name} ({len(scheme_rows):,} rows)")
