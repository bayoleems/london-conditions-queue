"""Model population, targets and features for the condition sign-off models.

Every feature is known on the day an application is submitted; look-back features use only
earlier applications. Both model notebooks build their tables from here, so they match exactly.

    from conditions_queue.features import load_dataset, add_jev_features, CAT, NUM, NUM_JEV
    d = load_dataset()
    d, coverage = add_jev_features(d)
"""
import json
import re

import numpy as np
import pandas as pd

from conditions_queue.paths import DISCHARGE_CSV, JEV_JSONL, PARQUET

STAT_DAYS = 56  # 8-week statutory period for approval of details (DMPO 2015 art. 27)

TOPICS = ["Accessibility", "Archaeology & heritage", "Boundary & lighting", "Construction logistics / management",
          "Contamination & ground", "Drainage & flood", "Ecology & biodiversity", "Energy & carbon", "Fire safety",
          "Highways & travel", "Landscaping & trees", "Materials & design detail", "Noise, vibration & air",
          "Piling & structure", "Secured by Design", "Waste, cycle & parking"]
# Topics usually attached as pre-commencement conditions (work can't start until they're cleared)
PRE_START = ["Construction logistics / management", "Contamination & ground", "Archaeology & heritage",
             "Drainage & flood", "Piling & structure", "Landscaping & trees"]


def slug(topic):
    """Keyword topic column name, e.g. 'Drainage & flood' -> 't_drainage_flood'."""
    return "t_" + re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


# ---------------------------------------------------------------- feature lists
TOPIC_COLS = [slug(t) for t in TOPICS]
CAT = ["area_name", "app_type", "app_size", "band"]
NUM = ["n_cond_c", "n_topics", "any_precommencement", "log_homes", "homes_known", "has_parent", "lag_days",
       "month", "dow", "desc_words", "has_agent", "boro_workload_28d", "boro_late_rate_365d",
       "scheme_prior_apps", "scheme_prior_refusal"] + TOPIC_COLS

# Description features: the keyword version (above) and Jev's version (add_jev_features)
KEYWORD_TEXT_COLS = TOPIC_COLS + ["n_cond_c", "n_topics", "any_precommencement"]
JEV_TOPIC_COLS = ["jev_" + slug(t)[2:] for t in TOPICS]
JEV_COLS = JEV_TOPIC_COLS + ["jev_n_cond_c", "jev_n_topics", "jev_topic_stated", "jev_precommencement"]
NUM_JEV = [c for c in NUM if c not in KEYWORD_TEXT_COLS] + JEV_COLS

# ---------------------------------------------------------------- chart labels
# Plain-English name followed by the real column name, so every chart traces back to the data
# (see docs/DATA_DICTIONARY.md).
_PLAIN = {
    "area_name": "Borough", "app_type": "Application type", "app_size": "Application size",
    "band": "Scheme size band", "n_cond_c": "Conditions bundled", "n_topics": "Number of topics",
    "any_precommencement": "Has a pre-start condition", "log_homes": "Number of homes", "homes_known": "Homes known",
    "has_parent": "Permission matched", "lag_days": "Days since permission", "month": "Month submitted",
    "dow": "Day of week submitted", "desc_words": "Description length", "has_agent": "Agent named",
    "boro_workload_28d": "Borough workload, last 4 weeks", "boro_late_rate_365d": "Borough late rate, last year",
    "scheme_prior_apps": "Scheme's earlier sign-offs", "scheme_prior_refusal": "Scheme had a refusal",
}
LABELS = {col: f"{name} ({col})" for col, name in _PLAIN.items()}
LABELS.update({c: f"{t}, keyword ({c})" for c, t in zip(TOPIC_COLS, TOPICS)})
LABELS.update({c: f"{t}, Jev ({c})" for c, t in zip(JEV_TOPIC_COLS, TOPICS)})
LABELS.update({
    "topics": "Condition topics (16 t_* flags)",
    "jev_topics": "Condition topics, Jev (16 jev_* flags)",
    "jev_n_cond_c": "Conditions bundled, Jev (jev_n_cond_c)",
    "jev_n_topics": "Number of topics, Jev (jev_n_topics)",
    "jev_topic_stated": "Topic stated in request, Jev (jev_topic_stated)",
    "jev_precommencement": "Says pre-start, Jev (jev_precommencement)",
})


# ---------------------------------------------------------------- dataset
def load_dataset():
    """One row per discharge application submitted 2021-2025 and not withdrawn.

    Targets:
      late      1 if decided after 56 days, or still undecided at fetch and older than 56 days
      days      days to decision; for undecided apps, days from submission to fetch (censored)
      decided   1 if a decision was made (event observed), 0 if still undecided (censored)
    """
    dis_all = pd.read_csv(DISCHARGE_CSV, parse_dates=["start_date", "decided_date", "parent_date"])
    raw = pd.read_parquet(PARQUET, columns=["name", "app_type", "app_size", "agent_company", "description",
                                            "fetched_at", "lat", "lng", "ward_name"])
    d = dis_all.merge(raw, on="name", how="left")
    d["fetched_at"] = pd.to_datetime(d["fetched_at"]).dt.tz_localize(None)

    d = d[d["outcome"].isin(["approved", "refused", "partial", "pending"]) & (d["start_date"].dt.year >= 2021)].copy()
    age = (d["fetched_at"] - d["start_date"]).dt.days
    is_decided = d["outcome"] != "pending"
    d = d[(is_decided & (d["days_to_decision"] >= 0)) | (~is_decided & (age > STAT_DAYS))].copy()

    d["decided"] = (d["outcome"] != "pending").astype(int)
    d["days"] = np.where(d["decided"] == 1, d["days_to_decision"], (d["fetched_at"] - d["start_date"]).dt.days)
    d["late"] = np.where(d["decided"] == 0, 1, (d["days_to_decision"] > STAT_DAYS).astype(int))

    topic_lists = d["topics"].fillna("").str.split("|")
    for t in TOPICS:
        d[slug(t)] = topic_lists.apply(lambda s: t in s).astype(int)
    d["n_topics"] = d[TOPIC_COLS].sum(axis=1)
    d["any_precommencement"] = d[[slug(t) for t in PRE_START]].max(axis=1)
    d["log_homes"] = np.log1p(d["homes"])
    d["homes_known"] = d["homes"].notna().astype(int)
    d["n_cond_c"] = d["n_cond"].clip(upper=10)
    d["has_parent"] = d["parent_uid"].notna().astype(int)
    d["lag_days"] = d["lag"].clip(upper=3650)
    d["month"] = d["start_date"].dt.month
    d["dow"] = d["start_date"].dt.dayofweek
    d["desc_words"] = d["description"].fillna("").str.split().str.len()
    d["has_agent"] = (d["agent_company"].fillna("") != "").astype(int)
    for c in ["app_type", "app_size"]:
        d[c] = d[c].replace("", "Unknown").fillna("Unknown")
    d["band"] = d["band"].fillna("Unknown")

    _add_look_back_features(d, dis_all)
    return d.reset_index(drop=True)


def _add_look_back_features(d, dis_all):
    """Borough workload and late rate, and the scheme's history, using only earlier applications."""
    by_borough = {b: g for b, g in dis_all.groupby("area_name")}
    by_scheme = {p: g for p, g in dis_all[dis_all["parent_uid"].notna()].groupby("parent_uid")}
    four_weeks, one_year = pd.Timedelta(days=28), pd.Timedelta(days=365)
    workload, late_rate, prior_apps, prior_refusal = [], [], [], []
    for r in d.itertuples():
        g, s = by_borough[r.area_name], r.start_date
        workload.append(((g["start_date"] < s) & (g["start_date"] >= s - four_weeks)).sum())
        past = g[(g["decided_date"] < s) & (g["decided_date"] >= s - one_year) & g["days_to_decision"].notna()]
        late_rate.append((past["days_to_decision"] > STAT_DAYS).mean() if len(past) >= 20 else np.nan)
        scheme = by_scheme.get(r.parent_uid) if isinstance(r.parent_uid, str) else None
        if scheme is None:
            prior_apps.append(0)
            prior_refusal.append(0)
        else:
            prior_apps.append((scheme["start_date"] < s).sum())
            prior_refusal.append(int(((scheme["decided_date"] < s) & scheme["outcome"].isin(["refused", "partial"])).any()))
    d["boro_workload_28d"], d["boro_late_rate_365d"] = workload, late_rate
    d["scheme_prior_apps"], d["scheme_prior_refusal"] = prior_apps, prior_refusal


def add_jev_features(d):
    """Attach Jev's description labels (outputs/jev_topics.jsonl).

    Applications Jev hasn't labelled fall back to the keyword condition count and get no topics.
    Returns (d, share of rows with a Jev label).
    """
    with open(JEV_JSONL) as f:
        jev = pd.DataFrame([json.loads(line) for line in f]).drop_duplicates("name", keep="last")
    d = d.drop(columns=[c for c in d.columns if c.startswith("jev_")]).merge(
        jev.rename(columns={"topics": "jev_topics", "n_conditions": "jev_n_conditions",
                            "topic_stated": "jev_stated_raw", "pre_commencement": "jev_pre_raw"}),
        on="name", how="left")
    labelled = d["jev_topics"].notna()
    topic_lists = d["jev_topics"].apply(lambda ts: ts if isinstance(ts, list) else [])
    for c, t in zip(JEV_TOPIC_COLS, TOPICS):
        d[c] = topic_lists.apply(lambda ts: int(t in ts))
    d["jev_n_cond_c"] = d["jev_n_conditions"].fillna(d["n_cond"]).clip(upper=10)
    d["jev_n_topics"] = d[JEV_TOPIC_COLS].sum(axis=1)
    d["jev_topic_stated"] = d["jev_stated_raw"].fillna(False).astype(int)
    d["jev_precommencement"] = d["jev_pre_raw"].map({True: 1, False: 0})
    return d, labelled.mean()


# ---------------------------------------------------------------- evaluation helpers
def time_split(d, cutoff="2024-01-01"):
    """Train on applications submitted before the cutoff, test on those from it onwards."""
    return d[d["start_date"] < cutoff].copy(), d[d["start_date"] >= cutoff].copy()


def concordance(days, decided, pred_days):
    """Harrell's C: share of comparable pairs where the application predicted to take longer really did.

    A pair (i, j) is comparable when i was decided and took less time than j (j may be undecided).
    """
    days, decided, pred = map(np.asarray, (days, decided, pred_days))
    conc = ties = total = 0.0
    for start in range(0, len(days), 500):
        i = slice(start, start + 500)
        di, ei, pi = days[i][:, None], decided[i][:, None], pred[i][:, None]
        comparable = (ei == 1) & (di < days[None, :])
        total += comparable.sum()
        conc += (comparable & (pi < pred[None, :])).sum()
        ties += (comparable & (pi == pred[None, :])).sum()
    return (conc + 0.5 * ties) / total
