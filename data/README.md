# Data

The raw and external data aren't committed: they're large, and some are third-party extracts. Put them here to rerun the pipeline. Everything derived from them is already in `outputs/`, so the notebooks and the site run without them.

```
data/
├── raw/
│   ├── foundations_london_housing_2015_2025_20261005T192558Z.parquet   # 31 MB
│   └── hpm_la_2024.zip                                                  # 1.2 GB
└── external/
    └── planning_data/
        ├── conservation-area/conservation-area.csv
        ├── article-4-direction-area/article-4-direction-area.csv
        ├── tree-preservation-zone/tree-preservation-zone.csv
        ├── listed-building/listed-building.csv
        └── brownfield-land/brownfield-land.csv
```

| File | What it is | Where to get it |
|---|---|---|
| `foundations_london_housing_*.parquet` | London planning applications 2015–2025, already filtered to housing, scraped by [PlanIt](https://www.planit.org.uk/) (243,926 rows, fetched August 2026) | Provided to House London teams. Any similar PlanIt extract with the same columns works ([PlanIt API](https://www.planit.org.uk/api/)) |
| `hpm_la_2024.zip` | House price per square metre, Price Paid linked to EPC floor areas, one CSV per local authority, to 2024 | [London Datastore: House price per square metre in England and Wales](https://data.london.gov.uk/dataset/house-price-per-square-metre-in-england-and-wales) |
| `planning_data/<layer>/<layer>.csv` | National bulk downloads of five constraint layers, fetched November 2026 | `https://files.planning.data.gov.uk/dataset/<layer>.csv` from [planning.data.gov.uk](https://www.planning.data.gov.uk/) |

The column-by-column description of the raw extract is in [`docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).
