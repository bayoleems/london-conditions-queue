# London's Conditions Queue

**Planning permission isn't permission to build.** Most permissions come with conditions (drainage, construction logistics, materials), and many must be signed off by the council before work can start. Councils have 8 weeks to decide each sign-off. In London, more than half take longer.

This project measures that queue and builds models that predict, on the day a sign-off is submitted, whether it will be late and how long it will take.

- **Dashboard:** https://bayoleems.github.io/london-conditions-queue/
- **Findings page:** https://bayoleems.github.io/london-conditions-queue/insight.html

## Key findings

From 16,284 discharge-of-condition applications on London housing schemes, submitted 2021–2025:

| | |
|---|---|
| Decided after the 8-week deadline | **56%** (54% of decided applications) |
| Median time to a decision | **60 days** |
| Best model for "will it be late?" (2024–25 test set) | **AUC 0.636**, against 0.595 for the borough average |
| Late sign-offs found by checking the riskiest 30% | **37%** |
| "Plan for" (P90) date met | **93%** of decisions |
| Each extra condition bundled into one application | about **9% slower** decision |

- **The description matters.** Without description features, the late model barely beats the borough average (0.596). Reading the conditions being signed off lifts it to 0.636.
- **Jev reads descriptions better than keywords.** Most descriptions repeat the original permission's text, so keyword matching tags the scheme rather than the condition. Labels from Jev (TypeSafe's decision model on OpenRouter) improve every model.
- **Decision time can't be predicted to the day.** Most decisions land near day 56 whatever the application, so the useful outputs are a ranking and a "plan for" range.
- **Biggest predictors:** application size (`app_size`), borough, what the conditions are about, and how many are bundled together.

## Repository layout

```
├── src/conditions_queue/      Python package: pipeline, features, labelling, checks
│   ├── paths.py               every file location in one place
│   ├── analysis.py            discharge applications, keyword topics, parent matching, summary
│   ├── features.py            model population, targets, features, chart labels
│   ├── classify_jev.py        description labels from Jev (OpenRouter decisions API)
│   ├── external_*.py          house prices and planning.data.gov.uk constraint features
│   ├── build_dashboard_data.py, build_insight_page.py
│   ├── checks/                independent checks of the deadline and extension figures
│   └── templates/             findings page template
├── notebooks/
│   ├── 01_conditions_queue_walkthrough.ipynb   raw extract to findings, step by step
│   ├── 02_late_model.ipynb                     will a sign-off be late?
│   ├── 03_days_model.ipynb                     how long will it take? what drives it?
│   ├── 04_description_topics.ipynb             what the descriptions say; keywords vs Jev
│   └── builders/                               scripts that generate the notebooks
├── outputs/                   derived tables (committed)
├── site/                      static dashboard, deployed to GitHub Pages
├── docs/                      data dictionary and the House London data asset register
└── data/                      raw and external data (not committed; see data/README.md)
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"
```

Everything in `outputs/` is committed, so the notebooks and the site run straight away. To rebuild from scratch, put the raw data in `data/` as described in [`data/README.md`](data/README.md), then run:

```bash
python -m conditions_queue.analysis                 # outputs/summary.json, discharge_apps.csv, schemes.csv
python -m conditions_queue.external_house_prices    # outputs/ext_house_prices.csv
python -m conditions_queue.external_constraints     # outputs/ext_constraints.csv
python -m conditions_queue.classify_jev             # outputs/jev_topics.jsonl (needs OPENROUTER_API_KEY, ~$1.40)
```

Then rebuild and run the notebooks, and refresh the site:

```bash
for b in notebooks/builders/make_0*.py; do python "$b"; done
(cd notebooks && for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done)
python -m conditions_queue.build_dashboard_data     # outputs/ui_dashboard.json and site/data/
python -m conditions_queue.build_insight_page       # site/insight.html
```

Preview the site locally with `python -m http.server --directory site` and open http://localhost:8000.

### API key

`classify_jev` reads `OPENROUTER_API_KEY` from the environment, or from a `.env` file at the repo root containing `OPENROUTER_API_KEY=...`. `.env` is ignored by git.

## Method in brief

- **Population:** applications to discharge planning conditions on housing schemes, picked out of the description text; variations, non-material amendments and reserved matters are excluded.
- **Targets:** `late` (decided after 56 days, or still undecided past that) and `days` (submission to decision, with undecided applications treated as censored).
- **Features:** only what is known on the day of submission. Look-back features use earlier applications only.
- **Split:** train on 2021–2023 submissions, test on 2024–2025.
- **Models:** logistic regression and LightGBM for late or not; ridge, LightGBM, quantile LightGBM and a Cox survival model for decision time.

Every column, target and feature is defined in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Limitations

- These are associations, not causes.
- Consultee responses, extensions of time and validation dates aren't in the data, and are likely the biggest missing drivers.
- Coverage is uneven: several boroughs, including Barnet, Tower Hamlets, Westminster and Newham, are barely in the extract.
- How PlanIt assigns `app_size` hasn't been confirmed from its documentation; it tracks scheme size closely.

## Data sources

- Planning applications: [PlanIt](https://www.planit.org.uk/), via a House London extract
- Site constraints: [planning.data.gov.uk](https://www.planning.data.gov.uk/)
- House price per square metre: [GLA London Datastore](https://data.london.gov.uk/dataset/house-price-per-square-metre-in-england-and-wales)
- Description labels: Jev by TypeSafe, through [OpenRouter](https://openrouter.ai/)
