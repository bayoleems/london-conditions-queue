"""Build the "London's Conditions Queue" findings page: inject the summary figures into the page template.

Adds `by_topic_jev` to outputs/summary.json's figures: the same late and refusal rates as `by_topic`,
but with topics labelled by Jev (outputs/jev_topics.jsonl) from only the conditions being signed off.

Run: python -m conditions_queue.build_insight_page   ->  site/insight.html
"""
import json

import pandas as pd

from conditions_queue.paths import DISCHARGE_CSV, JEV_JSONL, SITE_INSIGHT_HTML, SUMMARY_JSON, TEMPLATES

MIN_TOPIC_N = 200  # same threshold as by_topic in analysis.py


def topic_summary_jev():
    """Late and refusal rates per Jev topic, with the same population and definitions as analysis.py's by_topic."""
    apps = pd.read_csv(DISCHARGE_CSV, parse_dates=["start_date"])
    with open(JEV_JSONL) as f:
        jev = pd.DataFrame([json.loads(line) for line in f])[["name", "topics"]]
    apps = apps.merge(jev.rename(columns={"topics": "jev_topics"}), on="name", how="left")
    eligible = apps[(apps["outcome"] != "withdrawn") & (apps["start_date"].dt.year >= 2021)].copy()
    eligible["late"] = eligible["over8"].astype(str) == "True"
    decided = eligible["outcome"].isin(["approved", "refused", "partial"])
    topic_lists = eligible["jev_topics"].apply(lambda ts: ts if isinstance(ts, list) else [])

    out = {}
    for topic in sorted({t for ts in topic_lists for t in ts}):
        has = topic_lists.apply(lambda ts: topic in ts)
        if has.sum() < MIN_TOPIC_N:
            continue
        resolved = eligible[has & decided]
        out[topic] = {
            "n": int(has.sum()),
            "median_days": float(resolved["days_to_decision"].median()),
            "pct_over_8wk": round(100 * eligible.loc[has, "late"].mean(), 1),
            "pct_not_approved": round(100 * resolved["outcome"].isin(["refused", "partial"]).mean(), 1),
        }
    return out


def main():
    data = json.loads(SUMMARY_JSON.read_text())
    data["by_topic_jev"] = topic_summary_jev()
    template = (TEMPLATES / "insight_page.html").read_text()
    SITE_INSIGHT_HTML.parent.mkdir(parents=True, exist_ok=True)
    SITE_INSIGHT_HTML.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"wrote {SITE_INSIGHT_HTML} ({len(data['by_topic_jev'])} Jev topics)")


if __name__ == "__main__":
    main()
