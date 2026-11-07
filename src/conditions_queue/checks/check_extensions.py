"""Check the two open questions behind the 8-week deadline figures:

  A. Is "16 deemed discharges" a real signal, or just how a few councils label decisions?
  B. Can agreed extensions of time be told apart from genuinely late decisions?

Reads the raw parquet extract directly.
Run: python -m conditions_queue.checks.check_extensions
"""
import pandas as pd

from conditions_queue.paths import PARQUET as SRC
STAT_DAYS = 56

# Discharge-of-conditions filter, same rules as analysis.py
DISCHARGE = (r"details (?:pursuant|of|submitted|required|in respect)|discharge|approval of details|"
             r"submission of details|pursuant to condition|required by condition")
NOT_DISCHARGE = (r"variation|vary|removal of condition|remove condition|section 73|\bs\.?73|"
                 r"minor material|non[- ]material|reserved matters")
RESIDENTIAL = (r"dwelling|\bflats?\b|residential|\bhomes?\b|\bhouses?\b|\bunits?\b|\bC3\b|apartment|"
               r"maisonette|\bHMO\b")

raw = pd.read_parquet(SRC)
for c in ("start_date", "decided_date", "target_decision_date"):
    raw[c] = pd.to_datetime(raw[c])
decision = raw["decision"].fillna("")
desc = raw["description"].fillna("")

dis = raw[desc.str.contains(DISCHARGE, case=False)
          & ~desc.str.contains(NOT_DISCHARGE, case=False)
          & desc.str.contains(RESIDENTIAL, case=False)].copy()
dis_decision = dis["decision"].fillna("")
decided = dis[dis["decided_date"].notna() & (dis["days_to_decision"] >= 0)
              & (dis["start_date"].dt.year >= 2021)
              & ~dis_decision.str.contains("withdraw", case=False)].copy()
print(f"{len(raw):,} rows · {len(dis):,} discharge applications · {len(decided):,} decided 2021-2025\n")

# ------------------------------------------------------------------ A. deemed discharge
print("A. Deemed discharge: where is it recorded?")
is_deemed = decision.str.contains("deemed", case=False) & decision.str.contains(r"discharge|AOD", case=False)
deemed = raw[is_deemed]
late_by_boro = decided[decided["days_to_decision"] > STAT_DAYS].groupby("area_name").size()
tab = pd.DataFrame({
    "deemed_discharges": deemed.groupby("area_name").size(),
    "late_discharge_decisions": late_by_boro,
}).fillna(0).astype(int)
tab["deemed_per_100_late"] = (100 * tab["deemed_discharges"] / tab["late_discharge_decisions"]).round(1)
tab = tab.sort_values(["deemed_discharges", "late_discharge_decisions"], ascending=False)
print(tab.head(12).to_string())
n_boro = (tab["deemed_discharges"] > 0).sum()
print(f"\n  Boroughs recording any deemed discharge: {n_boro} of {len(tab)}")
print("  Labels used:", deemed["decision"].value_counts().to_dict())
print("  Is it in our discharge filter too?", int(is_deemed.loc[dis.index].sum()), "of", len(deemed))

# ------------------------------------------------------------------ B. extensions of time
print("\nB1. Decision labels mentioning 'extension' are NOT extensions of time")
ext = raw[decision.str.contains(r"extension|EOT|extended", case=False)]
print(ext["decision"].value_counts().to_string())
print(f"  Of these, discharge applications: {ext.index.isin(dis.index).sum()}")

print("\nB2. target_decision_date: does it move when an extension is agreed?")
decided["target_days"] = (decided["target_decision_date"] - decided["start_date"]).dt.days
has_t = decided["target_days"].notna()
print(f"  Decided apps with a target date: {has_t.sum():,} of {len(decided):,} ({100 * has_t.mean():.0f}%)")
print("  Coverage by borough (share with a target date):")
print(decided.groupby("area_name")["target_days"].apply(lambda s: s.notna().mean()).mul(100).round(0)
      .sort_values().to_string())

t = decided[has_t].copy()
t["target_band"] = pd.cut(t["target_days"], [-10_000, 55, 56, 10_000],
                          labels=["< 56 days", "exactly 56", "> 56 (possible extension)"])
print("\n  Target period relative to start date:")
print(t["target_band"].value_counts().sort_index().to_string())

late_56 = t["days_to_decision"] > STAT_DAYS
late_target = t["decided_date"] > t["target_decision_date"]
print("\n  Among decided apps with a target date:")
print(f"    Late vs fixed 8 weeks:          {100 * late_56.mean():.1f}%")
print(f"    Late vs their own target date:  {100 * late_target.mean():.1f}%")
print(f"    Late at 8 weeks but on time vs a later target (likely extension): "
      f"{(late_56 & ~late_target).sum():,} ({100 * (late_56 & ~late_target).mean():.1f}%)")

print("\n  Same comparison by borough (boroughs with 300+ decided apps with a target date):")
g = t.assign(late_56=late_56, late_target=late_target).groupby("area_name")
by = pd.DataFrame({"n": g.size(), "late_vs_8wk_%": g["late_56"].mean().mul(100).round(1),
                   "late_vs_target_%": g["late_target"].mean().mul(100).round(1)})
by["gap"] = (by["late_vs_8wk_%"] - by["late_vs_target_%"]).round(1)
print(by[by["n"] >= 300].sort_values("gap", ascending=False).to_string())

print("\n  Sanity check: decisions just after the target date (if targets were moved after the fact,")
print("  you'd expect many decisions landing exactly on the target)")
offset = (t["decided_date"] - t["target_decision_date"]).dt.days
print(offset.clip(-14, 14).value_counts().sort_index().loc[-7:7].to_string())
