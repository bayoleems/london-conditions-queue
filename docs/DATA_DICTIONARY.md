# Data dictionary: Conditions Queue models

This covers every table the condition sign-off analysis reads or writes: the raw extract, the derived application table, the model targets and features, the Jev labels, the external data, and the exports behind the dashboard.

**Population used by the models:** discharge-of-condition applications on housing schemes, submitted 2021–2025 and not withdrawn. That's **16,284** applications (17,079 before those filters).

**Rule for model features:** a feature may only use information known on the day the application is submitted. Columns that are only known later are marked *leak* below and are never used as features.

---

## 1. Raw extract: `data/raw/foundations_london_housing_2015_2025_20261005T192558Z.parquet`

A PlanIt-derived extract of London planning applications, already filtered to housing-relevant applications. It has 243,926 rows, one per application, with submission dates from November 2015 to December 2025, fetched 4–11 August 2026.

| Column | Type | Meaning | Used as |
|---|---|---|---|
| `name` | text | PlanIt's unique application id: borough + council reference, e.g. `Barnet/24/2686/OUT` | Join key everywhere |
| `area_name` | text | Planning authority as PlanIt names it, e.g. `Kensington` = Kensington and Chelsea, `City` = City of London, `Old Oak Park Royal` and `London Legacy` = development corporations | Feature: **Borough** |
| `reference` | text | Council's reference (often empty) | Not used |
| `uid` | text | Council's own reference as PlanIt stores it | Matching to the parent permission (in `conditions_queue.analysis`) |
| `url` | text | Link to the application on the council's portal | Not used |
| `description` | text | Free text describing the application. For discharge applications: the **request** (which conditions) followed by a **copy of the original permission's description** | Source of all description features (keyword and Jev) |
| `app_type` | text | PlanIt's type: Full, Outline, Conditions, Amendment, Heritage, Trees, Other | Feature: **Application type** |
| `app_size` | text | PlanIt's size label: Small, Medium, Large. How PlanIt sets it hasn't been verified | Feature: **Application size**. One of the two strongest predictors |
| `status` | text | PlanIt's status: Permitted, Conditions, Rejected, Undecided, Withdrawn, Unresolved, Referred | Builds `outcome`. *Leak* as a feature |
| `decision` | text | The council's decision wording, e.g. "Approve with Conditions", "Refuse", "Deemed Discharge (AOD)" | Builds `outcome`, deemed-discharge count. *Leak* |
| `decided_by` | text | Delegated, committee and so on | *Leak*, not used |
| `start_date` | date | Date the application was registered or validated. **Day 0 of the 8-week clock** | Time split; month and weekday features; look-back features |
| `decided_date` | date | Date of the decision | Builds `days`. *Leak* |
| `target_decision_date` | date | The council's target date. Some boroughs move it when an extension of time is agreed; nine boroughs never fill it | Extension analysis only (`conditions_queue.checks.check_extensions`). *Leak* |
| `days_to_decision` | number | `decided_date − start_date` in days (matches the two dates on 100% of rows) | Builds `days` and `late` |
| `last_changed` | timestamp | When PlanIt last saw a change | *Leak*, not used |
| `fetched_at` | timestamp | When the row was scraped (4–11 Aug 2026) | Days-so-far for undecided applications |
| `n_comments` | number | Public comments received | *Leak*: builds up during the application |
| `n_documents` | number | Documents on the file | *Leak*: builds up during the application |
| `n_statutory_days` | number | PlanIt's statutory period for the application type (mostly 56) | Not used |
| `n_dwellings` | number | Number of homes where PlanIt recorded it (11k rows) | Fallback for `homes` |
| `housing_relevance_score` | number | Score from the extract's housing filter (0.9–1.0) | Not used |
| `lat`, `lng` | number | Site location (95% of discharge applications) | Spatial join to site constraints |
| `ward_name` | text | Electoral ward | Not used |
| `agent_company` | text | Planning agent, blank for 55% of discharge applications | Feature: **Agent named** (yes/no) |
| `case_officer` | text | Empty in this extract | Not used |

---

## 2. Derived application table: `outputs/discharge_apps.csv`

Written by `analysis.py`. One row per discharge-of-condition application (17,079 rows).

| Column | Meaning |
|---|---|
| `name`, `area_name`, `start_date`, `decided_date`, `days_to_decision` | As in the raw extract |
| `outcome` | `approved`, `refused`, `partial`, `pending` (undecided), `withdrawn`, `other`, from `decision` and `status` |
| `over8` | True if decided after 56 days, or undecided and older than 56 days at the end of 2025 (the page's definition) |
| `band` | Scheme size: Minor (1–9 homes), Major (10–149), Strategic (150+), Unknown |
| `homes` | Number of homes, read from the description or the parent permission |
| `n_cond` | Keyword count of conditions in the request |
| `topics` | Keyword topics, `|`-separated (see section 4) |
| `parent_uid` | The original permission, matched by reference number (33% matched) |
| `parent_date` | Decision date of the original permission, from the match or a "dated …" phrase |
| `lag` | Days from the original permission to this application |

**How an application is picked out as discharge-of-condition:** the description mentions approval or discharge of details under a condition, isn't a variation (section 73), non-material amendment or reserved matters, and mentions homes.

---

## 3. Targets

Built by `conditions_queue.features.load_dataset()`.

| Target | Type | Definition | Used by |
|---|---|---|---|
| `late` | 0/1 | 1 if decided more than **56 days** after `start_date`, or still undecided when fetched (all of those are past 56 days). 56% of the 16,284 | Late models (logistic regression, LightGBM) |
| `days` | number | Decided: `days_to_decision`. Undecided: days from `start_date` to `fetched_at` (a lower bound) | Days models |
| `decided` | 0/1 | 1 if decided, 0 if still undecided (the event flag for survival analysis). 854 are undecided | Cox survival model, C-index |

How the days models use the target: ridge and LightGBM predict **log(1 + days)** for decided applications. The Cox survival model uses `days` with `decided` as the event flag. The quantile LightGBM models predict days directly at the 50th (P50) and 90th (P90) percentiles.

**Time split:** train on `start_date` before 1 January 2024 (10,705 applications), test on 2024–2025 (5,579).

---

## 4. Model features

Built by `conditions_queue/features.py`. All are known on the day of submission.

### Application and timing

| Feature | Type | Meaning |
|---|---|---|
| `area_name` | category | Borough |
| `app_type` | category | PlanIt application type |
| `app_size` | category | PlanIt size label (Small / Medium / Large / Unknown) |
| `band` | category | Scheme size band from the number of homes |
| `log_homes` | number | log(1 + homes); blank when unknown |
| `homes_known` | 0/1 | Whether the number of homes is known |
| `has_parent` | 0/1 | Whether the original permission was matched |
| `lag_days` | number | Days since the original permission (capped at 3,650) |
| `month`, `dow` | number | Month and weekday of submission |
| `desc_words` | number | Words in the description |
| `has_agent` | 0/1 | A planning agent is named |

### Look-back features

These use only applications submitted or decided **before** this one.

| Feature | Meaning |
|---|---|
| `boro_workload_28d` | Discharge applications the borough received in the previous 28 days |
| `boro_late_rate_365d` | The borough's late share for decisions in the previous year (blank if under 20) |
| `scheme_prior_apps` | Earlier discharge applications on the same permission |
| `scheme_prior_refusal` | An earlier one on the same permission was refused or part-refused |

### Description features: keyword version (`analysis.py`)

These read the **whole** description, so they often pick up words from the copied permission text.

| Feature | Meaning |
|---|---|
| `t_<topic>` | 16 flags, one per topic below, set when the topic's keywords appear anywhere |
| `n_topics` | Number of topic flags set |
| `n_cond_c` | Keyword condition count (capped at 10) |
| `any_precommencement` | Has a topic that is usually a pre-start condition (inferred, not read) |

### Description features: Jev version (`classify_jev.py`)

These read only the **request**, using Jev's answers to typed questions.

| Feature | Meaning |
|---|---|
| `jev_<topic>` | 16 flags: Jev's probability that a requested condition covers the topic is 0.5 or more |
| `jev_n_topics` | Number of Jev topic flags set |
| `jev_n_cond_c` | Jev's condition count (capped at 10; falls back to the keyword count if Jev couldn't tell) |
| `jev_topic_stated` | The request says what the conditions cover, rather than only giving numbers |
| `jev_precommencement` | The request says the details are needed before work starts (1), at a later stage (0), or doesn't say (blank). Stated in under 1% of descriptions |

### How charts name features

Charts in the notebooks and on the dashboard use a plain-English name **followed by the column name in brackets**, set in `conditions_queue.features.LABELS`. For example:

| Chart label | Column |
|---|---|
| Application size (app_size) | `app_size` |
| Borough (area_name) | `area_name` |
| Application type (app_type) | `app_type` |
| Scheme size band (band) | `band` |
| Conditions bundled (n_cond_c) | `n_cond_c` (keyword count) |
| Conditions bundled, Jev (jev_n_cond_c) | `jev_n_cond_c` (Jev count) |
| Condition topics (16 t_* flags) | the 16 keyword topic flags, shuffled together as one group |
| Condition topics, Jev (16 jev_* flags) | the 16 Jev topic flags, shuffled together as one group |
| Description length (desc_words) | `desc_words` |
| Borough late rate, last year (boro_late_rate_365d) | `boro_late_rate_365d` |
| Application size: Large (app_size) | the `app_size = Large` level, in the Cox chart |

### The 16 topics

Both versions use the same definitions.

| Topic | Covers |
|---|---|
| Accessibility | M4(2) / M4(3), wheelchair housing |
| Archaeology & heritage | Archaeology, written scheme of investigation, heritage recording |
| Boundary & lighting | Boundary treatment, fences, walls, external lighting |
| Construction logistics / management | CLP, CMP, CEMP, demolition or construction method statements |
| Contamination & ground | Contamination, remediation, site investigation, verification reports |
| Drainage & flood | SuDS, surface or foul water drainage, flood risk, sewage |
| Ecology & biodiversity | Ecology, biodiversity, bat or bird boxes, green or brown roofs |
| Energy & carbon | Energy strategy, SAP, BREEAM, carbon, overheating, PV, heat pumps, water efficiency |
| Fire safety | Fire safety, fire strategy |
| Highways & travel | Highway works, s278, travel plans, delivery and servicing |
| Landscaping & trees | Landscaping, planting, tree protection, arboricultural statements |
| Materials & design detail | Materials, samples, brick, cladding, windows, balustrades, detailed drawings |
| Noise, vibration & air | Noise, sound insulation, vibration, air quality, dust, ventilation |
| Piling & structure | Piling, foundations, basements, structural details, site levels |
| Secured by Design | Secured by Design, crime prevention |
| Waste, cycle & parking | Refuse and cycle stores, car parking, EV charging, car clubs |

---

## 5. Jev labels: `outputs/jev_topics.jsonl`

Written by `classify_jev.py`, using Jev (`~typesafe/jev-latest`, served as `typesafe/jev-1.14-20261014`) through OpenRouter's decisions API. One JSON object per application, covering all 17,079. The run cost $1.41.

| Field | Type | Meaning |
|---|---|---|
| `name` | text | Application id |
| `topics` | list | Topics with probability ≥ 0.5 |
| `topic_probs` | object | Jev's probability (0–1) for each of the 16 topics |
| `n_conditions` | number or null | Number of conditions being discharged (10 = ten or more; null = can't tell) |
| `topic_stated` | true/false | The request names or describes what the conditions cover |
| `pre_commencement` | true/false/null | Needed before work starts / at a later stage / not stated |
| `model` | text | The Jev version that answered |

An earlier, partial run through `typesafe/jev-router` (12,560 rows, answered by DeepSeek) was replaced by this one and isn't in the repo.

---

## 6. External data

| File | Written by | Source | Columns |
|---|---|---|---|
| `outputs/ext_house_prices.csv` | `external_house_prices.py` | `data/raw/hpm_la_2024.zip`: house price per m², Price Paid × EPC linked (Data Asset Register row 36) | `area_name`, `month`, `ppm2_12m` (median £/m² over the last 12 months), `n_sales_12m`, `ppm2_yoy` (% change on a year earlier) |
| `outputs/ext_constraints.csv` | `external_constraints.py` | planning.data.gov.uk bulk CSVs in `data/external/planning_data/` (register row 24), fetched November 2026 | `name`, `in_conservation_area`, `in_article4_area`, `in_tpo_zone` (0/1, point inside the area), `listed_within_100m` (count), `brownfield_within_50m` (0/1). Only designations in force on the submission date count |

Article 4 and tree preservation zones are recorded too unevenly between boroughs to use. Flood risk zones weren't fetched, because the file is 2.5 GB.

---

## 7. Outputs

| File | Written by | Contents |
|---|---|---|
| `outputs/summary.json` | `analysis.py` | All figures on the London's Conditions Queue page |
| `site/insight.html` | `build_insight_page.py` | The findings page: `summary.json` plus `by_topic_jev`, the late and refusal rates per topic using Jev's labels (same population and definitions as `by_topic`) |
| `outputs/ui_late_model.json` | `notebooks/02_late_model.ipynb`, last section | Late-model scores, risk groups, importance, bundling, Jev comparison |
| `outputs/ui_days_model.json` | `notebooks/03_days_model.ipynb`, last section | Days scores, accuracy, predicted vs actual, ranges, Cox effects, importance, description test, external tests |
| `outputs/ui_dashboard.json` | `build_dashboard_data.py` | The two exports above reshaped into one table per dashboard chart (section 8); also copied to `site/data/` |

### Scores used throughout

| Score | Meaning |
|---|---|
| AUC | How well a model ranks applications from most to least likely to be late. 0.5 = coin flip, 1 = perfect |
| C-index | The share of application pairs put in the right order of time taken (undecided applications count as "at least this long") |
| MAE | Average absolute error in days, for decided applications |
| Rank correlation | Spearman correlation between predicted and actual days |
| Speed ratio | Cox hazard ratio: how a feature changes the daily chance of a decision. Below 1 = slower |
| Importance share | Drop in a model's test score when a feature is shuffled, as a share of all drops |

---

## 8. Dashboard tables (`outputs/ui_dashboard.json`)

Each key is one table, loaded by the Conditions Queue Models dashboard.

| Table | One row per | Columns |
|---|---|---|
| `headline` | (single row) | `n_applications`, `late_share`, `test_n`, `test_late_share`, `best_model`, `best_auc`, `borough_auc`, `no_text_auc`, `median_days`, `n_undecided`, `p90_coverage`, `per_condition_slowdown`, `per_condition_slowdown_jev`, `jev_coverage`, `borough_within_28`, `borough_median_error` |
| `late_scores` | model | `model`, `auc`, `accuracy`, `brier` |
| `risk_deciles` | risk group | `decile`, `applications`, `predicted`, `actual` |
| `bundling` | conditions bundled | `conditions`, `predicted_keyword`, `actual_keyword`, `predicted_jev`, `actual_jev`, `applications_jev` |
| `days_scores` | model | `model`, `c_index`, `mae_days`, `auc` |
| `days_accuracy` | model | `model`, `mean_error`, `median_error`, `within_14`, `within_28`, `rank_corr` |
| `days_calibration` | predicted-time group | `group`, `applications`, `predicted_median`, `actual_median`, `actual_p25`, `actual_p75` |
| `ranges` | conditions bundled | `conditions`, `applications`, `p50`, `p90`, `actual_median` |
| `cox_effects` | feature | `feature`, `ratio`, `low`, `high` |
| `importance` | method × feature | `method` (keyword / jev), `feature`, `avg_share`, `avg_rank`, `top5_models`, one share column per model |
| `description_value` | feature set × model | `features`, `model`, `metric`, `value`, `gain` (change against no description features) |
| `jev_model_scores` | model | `model`, `c_keyword`, `c_jev`, `auc_keyword`, `auc_jev` |
| `jev_feature_sets` | feature set | `features`, `logistic_auc`, `lightgbm_auc`, `lightgbm_sd` |
| `topic_rates` | topic | `topic`, `keyword_pct`, `jev_pct` |
| `external_tests` | source × feature set | `test`, `features`, `late_auc`, `days_c` |
| `constraints_within` | constraint | `constraint`, `late_with`, `late_without`, `diff_within_borough`, `boroughs` |

---

## 9. Code and notebooks

Modules are in `src/conditions_queue/` and run with `python -m conditions_queue.<module>` (see the README).

| File | What it does |
|---|---|
| `analysis.py` | Picks out discharge applications, keyword topics and counts, parent matching; writes `summary.json`, `discharge_apps.csv` and `schemes.csv` |
| `features.py` | Model population, targets, features, Jev features and chart labels |
| `classify_jev.py` | Labels descriptions with Jev (needs `OPENROUTER_API_KEY`, about $0.08 per 1,000) |
| `external_house_prices.py`, `external_constraints.py` | Build the external feature tables |
| `build_dashboard_data.py` | Builds `ui_dashboard.json` and copies it into the site |
| `build_insight_page.py` | Builds `site/insight.html` from `summary.json` |
| `checks/check_deadline.py`, `checks/check_extensions.py` | Independent checks of the 8-week, deemed-discharge and extension figures |
| `paths.py` | Every file location in one place |
| `notebooks/01_conditions_queue_walkthrough.ipynb` | Every step from the raw extract to the findings, cross-checked against `analysis.py` |
| `notebooks/02_late_model.ipynb` | Late-or-not models, and how Jev reads the descriptions |
| `notebooks/03_days_model.ipynb` | Decision-time models, cross-model importance, description test, external data |
| `notebooks/04_description_topics.ipynb` | What descriptions say; keyword tagger vs Jev |
| `notebooks/builders/` | Scripts that generate the four notebooks |
