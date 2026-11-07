"""Combine the two model notebooks' exports into one file for the Conditions Queue Models dashboard.

Inputs : outputs/ui_late_model.json (notebooks/02_late_model.ipynb, last section)
         outputs/ui_days_model.json (notebooks/03_days_model.ipynb, last section)
Output : outputs/ui_dashboard.json, copied to site/data/ui_dashboard.json for GitHub Pages.
         One key per table, each a list of row objects with plain column names (docs/DATA_DICTIONARY.md).

Run: python -m conditions_queue.build_dashboard_data
"""
import json
import shutil

from conditions_queue.paths import SITE_DATA_JSON, UI_DASHBOARD_JSON, UI_DAYS_JSON, UI_LATE_JSON

late = json.loads(UI_LATE_JSON.read_text())
days = json.loads(UI_DAYS_JSON.read_text())

MODEL_COLS = {"Late · logistic": "late_logistic", "Late · LightGBM": "late_lightgbm", "Days · ridge": "days_ridge",
              "Days · LightGBM": "days_lightgbm", "Days · Cox (survival)": "days_cox"}
r3 = lambda x: None if x is None else round(float(x), 4)

late_scores = [{"model": r["model"], "auc": r3(r["AUC"]), "accuracy": r3(r["Accuracy"]), "brier": r3(r["Brier"])}
               for r in late["scores"]]
jev_sets = late["jev"]["compare"]
for r in jev_sets:
    if r["description features"].startswith("JEV"):
        late_scores.append({"model": "LightGBM, JEV description features", "auc": r3(r["LightGBM AUC"]),
                            "accuracy": None, "brier": None})
best = max((r for r in late_scores if r["model"] != "Always 'late'"), key=lambda r: r["auc"])
borough = next(r for r in late_scores if r["model"] == "Borough late rate")

kw_bund = {r["conditions"]: r for r in late["bundling_keyword"]}
jev_bund = {r["conditions bundled"]: r for r in late["jev"]["bundling"]}
bundling = []
for k in sorted(set(kw_bund) | set(jev_bund)):
    a, b = kw_bund.get(k, {}), jev_bund.get(k, {})
    bundling.append({"conditions": int(k),
                     "predicted_keyword": r3(a.get("predicted late (all else equal)")),
                     "actual_keyword": r3(a.get("actually late (raw)")),
                     "predicted_jev": r3(b.get("predicted late (JEV count)")),
                     "actual_jev": r3(b.get("actually late (JEV count)")),
                     "applications_jev": b.get("applications")})


def importance(rows, method):
    out = []
    for r in rows:
        row = {"method": method, "feature": r["feature"], "avg_share": r3(r["average share %"]),
               "avg_rank": r3(r["average rank"]), "top5_models": int(r["top 5 in how many models"])}
        for name, col in MODEL_COLS.items():
            row[col] = r3(r.get(name))
        out.append(row)
    return sorted(out, key=lambda r: -r["avg_share"])[:12]


ds = days["days_summary"]
acc = {r["model"]: r for r in days["days_accuracy"]}
base_acc = acc["Borough median (baseline)"]
DV_METRICS = [("Late · logistic", "AUC", "late_logistic_auc"), ("Late · LightGBM", "AUC", "late_lightgbm_auc"),
              ("Days · ridge", "C-index", "days_ridge_c"), ("Days · LightGBM", "C-index", "days_lightgbm_c"),
              ("Days · Cox", "C-index", "days_cox_c"), ("Days · ridge", "MAE (days)", "days_ridge_mae"),
              ("Days · LightGBM", "MAE (days)", "days_lightgbm_mae")]
dv = {r["features"]: r for r in days["description_value"]}
none_row = dv["No description features"]
description_value = [{"features": f, "model": model, "metric": metric, "value": r3(r[col]), "gain": r3(r[col] - none_row[col])}
                     for f, r in dv.items() for model, metric, col in DV_METRICS]
headline = [{
    "n_applications": late["n_applications"], "late_share": r3(late["late_share"]),
    "test_n": late["test"]["n"], "test_late_share": r3(late["test"]["late_share"]),
    "best_model": best["model"], "best_auc": best["auc"], "borough_auc": borough["auc"],
    "median_days": ds["median_decided"], "n_undecided": ds["n_undecided"],
    "p90_coverage": r3(ds["p90_coverage"]),
    "per_condition_slowdown": r3(ds["per_condition_slowdown"]),
    "per_condition_slowdown_jev": r3(ds["per_condition_slowdown_jev"]),
    "jev_coverage": r3(late["jev"]["coverage"]),
    "no_text_auc": r3(none_row["late_lightgbm_auc"]),
    "borough_within_28": r3(base_acc["within 28 days"]),
    "borough_median_error": r3(base_acc["median error (days)"]),
}]

out = {
    "headline": headline,
    "late_scores": late_scores,
    "risk_deciles": [{"decile": int(r["decile"]), "applications": r["applications"],
                      "predicted": r3(r["predicted"]), "actual": r3(r["actually_late"])} for r in late["risk_deciles"]],
    "bundling": bundling,
    "days_scores": [{"model": r["model"], "c_index": r3(r["C-index"]), "mae_days": r3(r["MAE (days)"]),
                     "auc": r3(r["AUC (late)"])} for r in days["scores"]],
    "days_accuracy": [{"model": r["model"], "mean_error": r3(r["average error (days)"]), "median_error": r3(r["median error (days)"]),
                       "within_14": r3(r["within 14 days"]), "within_28": r3(r["within 28 days"]),
                       "rank_corr": r3(r["rank correlation"])} for r in days["days_accuracy"]],
    "days_calibration": [{"group": int(r["group"]), "applications": r["applications"], "predicted_median": r3(r["predicted_median"]),
                          "actual_median": r3(r["actual_median"]), "actual_p25": r3(r["actual_p25"]),
                          "actual_p75": r3(r["actual_p75"])} for r in days["days_calibration"]],
    "description_value": description_value,
    "ranges": [{"conditions": int(r["conditions"]), "applications": r["applications"], "p50": r3(r["P50"]),
                "p90": r3(r["P90"]), "actual_median": r3(r["actual_median"])} for r in days["ranges_by_conditions"]],
    "cox_effects": [{"feature": r["feature"], "ratio": r3(r["speed ratio"]), "low": r3(r["low"]), "high": r3(r["high"])}
                    for r in days["cox_effects"]],
    "importance": importance(days["importance_keyword"], "keyword") + importance(days["importance_jev"], "jev"),
    "jev_model_scores": [{"model": r["model"], "c_keyword": r3(r["C-index (keyword)"]), "c_jev": r3(r["C-index (JEV)"]),
                          "auc_keyword": r3(r["AUC (keyword)"]), "auc_jev": r3(r["AUC (JEV)"])} for r in days["jev_scores"]],
    "jev_feature_sets": [{"features": r["description features"], "logistic_auc": r3(r["Logistic AUC"]),
                          "lightgbm_auc": r3(r["LightGBM AUC"]), "lightgbm_sd": r3(r["LightGBM ± (5 seeds)"])} for r in jev_sets],
    "topic_rates": [{"topic": r["topic"], "keyword_pct": r3(r["keyword tagger %"]), "jev_pct": r3(r["JEV %"])}
                    for r in late["jev"]["topic_rates"]],
    "external_tests": (
        [{"test": "House prices per m²", "features": r["features"], "late_auc": r3(r["Late AUC"]),
          "days_c": r3(r["Days C-index"])} for r in days["external"]["house_prices"]]
        + [{"test": "Site constraints", "features": r["features"], "late_auc": r3(r["Late AUC"]),
            "days_c": r3(r["Days C-index"])} for r in days["external"]["constraints"]]),
    "constraints_within": [{"constraint": r["constraint"], "late_with": r3(r["late, with (%)"]),
                            "late_without": r3(r["late, without (%)"]),
                            "diff_within_borough": r3(r["difference within borough (points)"]),
                            "boroughs": int(r["boroughs compared"])} for r in days["external"]["constraints_within_borough"]],
}
UI_DASHBOARD_JSON.write_text(json.dumps(out, indent=1))
SITE_DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(UI_DASHBOARD_JSON, SITE_DATA_JSON)
print(f"wrote {UI_DASHBOARD_JSON.name} ({UI_DASHBOARD_JSON.stat().st_size:,} bytes) and copied it to site/data: "
      + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
