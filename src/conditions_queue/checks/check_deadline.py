"""Independent check of the 8-week deadline figures on the Conditions Queue page.

Re-derives every deadline claim straight from the parquet with pandas, without
reusing analysis.py, and prints each figure next to the value the page quotes.

Run: python -m conditions_queue.checks.check_deadline
"""
import re

import pandas as pd

from conditions_queue.paths import PARQUET as SRC
STAT_DAYS = 56  # 8 weeks: statutory period for approval of details (DMPO 2015 art. 27)

# Same discharge filter as analysis.py, restated here so the check stands alone
DISCHARGE = (r"details (?:pursuant|of|submitted|required|in respect)|discharge|approval of details|"
             r"submission of details|pursuant to condition|required by condition")
NOT_DISCHARGE = (r"variation|vary|removal of condition|remove condition|section 73|\bs\.?73|"
                 r"minor material|non[- ]material|reserved matters")
RESIDENTIAL = (r"dwelling|\bflats?\b|residential|\bhomes?\b|\bhouses?\b|\bunits?\b|\bC3\b|apartment|"
               r"maisonette|\bHMO\b")


def outcome(decision, status):
    d, s = str(decision or "").lower(), str(status or "").lower()
    if "withdraw" in d or s == "withdrawn":
        return "withdrawn"
    if "part" in d or "split" in d:
        return "partial"
    if re.search(r"refus|not approved|reject", d) or s == "rejected":
        return "refused"
    if s in ("permitted", "conditions") or re.search(r"grant|approv|agreed|discharg", d):
        return "approved"
    if s == "undecided" or not d:
        return "pending"
    return "other"


def check(label, got, page):
    print(f"  {label:<48} {got:>10}   (page: {page})")


raw = pd.read_parquet(SRC, columns=["uid", "area_name", "decision", "status", "start_date",
                                    "decided_date", "days_to_decision", "description"])
print(f"Source: {SRC.name} · {len(raw):,} rows\n")

desc = raw["description"].fillna("")
is_dis = (desc.str.contains(DISCHARGE, case=False, regex=True)
          & ~desc.str.contains(NOT_DISCHARGE, case=False, regex=True)
          & desc.str.contains(RESIDENTIAL, case=False, regex=True))
dis = raw[is_dis].copy()
dis["outcome"] = [outcome(d, s) for d, s in zip(dis["decision"], dis["status"])]
dis["start_date"] = pd.to_datetime(dis["start_date"])
dis["decided_date"] = pd.to_datetime(dis["decided_date"])
dis["year"] = dis["start_date"].dt.year

# Decided 2021-2025 submissions with a usable day count
dec = dis[dis["outcome"].isin(["approved", "refused", "partial"])
          & dis["days_to_decision"].notna() & (dis["days_to_decision"] >= 0)
          & (dis["year"] >= 2021)].copy()
days = dec["days_to_decision"]

print("1. Population")
check("Discharge applications (all years)", f"{len(dis):,}", "17,079")
check("Decided, submitted 2021-2025", f"{len(dec):,}", "n/a")

print("\n2. Is days_to_decision trustworthy? (recompute from the two dates)")
recomputed = (dec["decided_date"] - dec["start_date"]).dt.days
both = recomputed.notna()
match = (recomputed[both] == days[both]).mean() * 100
check("Rows where stored days == decided - start", f"{match:.1f}%", "should be ~100%")
late_recalc = (recomputed[both] > STAT_DAYS).mean() * 100
check("Late share using recomputed days", f"{late_recalc:.1f}%", "54%")

print("\n3. Deadline figures")
check("Decided after 8 weeks (>56 days)", f"{(days > STAT_DAYS).mean() * 100:.1f}%", "54%")
check("Median days to decision", f"{days.median():.0f}", "60")
check("Over six months (>182 days)", f"{(days > 182).mean() * 100:.1f}%", "11%")
check("Decided in final week (days 50-56)", f"{days.between(50, 56).mean() * 100:.1f}%", "22%")
check("Late decisions (count)", f"{(days > STAT_DAYS).sum():,}", "~8,300")

print("\n4. Deadline spike: decisions per week (week k = days 7k-6..7k)")
week = (-(-days // 7)).clip(upper=27).astype(int)
counts = week.value_counts().sort_index()
for k in range(5, 13):
    bar = "#" * int(counts.get(k, 0) / 50)
    flag = "  <- 8-week deadline" if k == 8 else ""
    print(f"  week {k:>2}: {counts.get(k, 0):>5,} {bar}{flag}")

print("\n5. Late share by submission year")
eligible = dis[(dis["outcome"] != "withdrawn") & (dis["year"] >= 2021)].copy()
age = (pd.Timestamp("2025-12-31") - eligible["start_date"]).dt.days
eligible["late"] = (eligible["days_to_decision"] > STAT_DAYS) | ((eligible["outcome"] == "pending") & (age > STAT_DAYS))
print(eligible.groupby("year")["late"].mean().mul(100).round(1).to_string())
print("  (page: 2021 ~64%, 2024-25 ~50%)")

print("\n6. Deemed discharge in decision labels (whole dataset)")
deemed = raw[raw["decision"].fillna("").str.contains("deemed", case=False)]
print(deemed["decision"].value_counts().to_string())
dd = deemed[deemed["decision"].str.contains(r"discharge|AOD", case=False)]
check("Deemed discharges of conditions", f"{len(dd):,}", "16")
print("  Deemed discharges by borough:")
print(dd["area_name"].value_counts().to_string())

print("\n7. Do any decision labels hint at extensions of time? (can't be separated otherwise)")
ext = raw["decision"].fillna("").str.contains(r"extension|EOT|extended", case=False)
print(f"  Decision labels mentioning extension: {ext.sum():,}")
