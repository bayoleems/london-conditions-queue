"""Builds notebooks/04_description_topics.ipynb: what the descriptions say, and Jev labels vs the keyword tagger.

Run from the repo root:
    python notebooks/builders/make_04_description_topics.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_description_topics.ipynb
Needs outputs/jev_topics.jsonl (src/conditions_queue/classify_jev.py).
"""
from pathlib import Path

import nbformat as nbf

NOTEBOOKS = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# What the descriptions say, and a better way to read them

Every discharge application has a free-text `description`. This notebook:
1. Shows what a description is made of.
2. Shows why the keyword tagger from `analysis.py` mislabels topics.
3. Compares it with labels from **Jev** (`~typesafe/jev-latest` through OpenRouter's decisions API), asked typed questions about only the conditions being signed off.
4. Rechecks the page's topic and bundling findings, and the late model, using JEV's labels.

Jev labels come from `src/conditions_queue/classify_jev.py` and are stored in `outputs/jev_topics.jsonl`. This notebook doesn't call the API.
""")

md("## 1 · Load")
code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))  # run from notebooks/: make src/ importable
from conditions_queue import paths

import json, re, textwrap, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from conditions_queue.features import CAT, NUM, TOPIC_COLS, TOPICS, slug, load_dataset, time_split

pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 200)
jev = pd.DataFrame([json.loads(l) for l in open(paths.OUTPUTS / "jev_topics.jsonl")]).drop_duplicates("name")
d = load_dataset()                                  # model population (2021-25, not withdrawn)
d = d.merge(jev.add_prefix("jev_").rename(columns={"jev_name": "name"}), on="name", how="left")
print(f"JEV labels: {len(jev):,} descriptions · matched to {d['jev_topics'].notna().mean():.1%} of the {len(d):,} modelled applications")
""")

md("""
## 2 · Anatomy of a description

A typical description has two parts:

| Part | Example | What it tells us |
|---|---|---|
| **The request** | "Approval of details pursuant to conditions 35 (Air Quality and Dust Management Plan), 48 (Construction and Environmental Management Plan)…" | Which conditions are being signed off, and usually what they cover |
| **The original permission, copied in** | "…of planning permission 19/02840/FUL (Demolition of the existing buildings and erection of a part 6 and part 7 storey building to provide 51 residential flats… cycle stores, car parking spaces, refuse/recycling stores, landscaping…)" | What the scheme is: homes, storeys, uses. **Not** what is being signed off |
""")
code("""
for r in d.sample(4, random_state=7).itertuples():
    print(textwrap.fill(r.description, 150)); print()
words = d["description"].str.split().str.len()
print(f"Median length {words.median():.0f} words · 90% are under {words.quantile(.9):.0f} words")
cond_no = d["description"].str.contains(r"condition[s]?\\s*\\(?\\d", case=False).mean()
perm_ref = d["description"].str.contains(r"\\d{2,4}/\\d{3,6}").mean()
print(f"Mentions a condition number: {cond_no:.0%}")
print(f"Mentions the original permission's reference: {perm_ref:.0%}")
""")

md("""
## 3 · The keyword tagger's problem

The tagger in `analysis.py` searches the **whole** description for words like "cycle" or "landscaping". But those words usually come from the copied scheme text ("…with cycle storage, refuse storage and landscaping"), not from the condition being signed off.
""")
code("""
has = d["jev_topics"].notna()
rows = []
for t in TOPICS:
    kw = d.loc[has, slug(t)].astype(bool)
    llm = d.loc[has, "jev_topics"].apply(lambda ts: t in ts)
    rows.append({"topic": t, "keyword tagger %": 100 * kw.mean(), "JEV %": 100 * llm.mean(),
                 "keyword tags JEV agrees with %": 100 * (kw & llm).sum() / max(kw.sum(), 1),
                 "JEV tags the keywords found %": 100 * (kw & llm).sum() / max(llm.sum(), 1)})
cmp_topics = pd.DataFrame(rows).set_index("topic").sort_values("keyword tagger %", ascending=False)
cmp_topics.round(0)
""")
code("""
fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(cmp_topics))
ax.barh(y + 0.2, cmp_topics["keyword tagger %"], height=0.4, color="#9aa3ae", label="Keyword tagger (whole description)")
ax.barh(y - 0.2, cmp_topics["JEV %"], height=0.4, color="#2a78d6", label="JEV (conditions requested only)")
ax.set_yticks(y); ax.set_yticklabels(cmp_topics.index, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("Share of applications tagged with the topic (%)")
ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
""")
md("""
**Reading this:** where the grey bar is much longer than the blue one, the keyword tagger is mostly tagging the **scheme**, not the condition. "Keyword tags JEV agrees with" is the share of keyword tags that JEV confirms. A low value means most of that topic's keyword tags are false alarms.
""")

md("""
### Spot check: disagreements

Read a few cases where the two disagree to judge which is right:
""")
code("""
dis = d[has].copy()
dis = dis[dis["jev_topics"].apply(len) > 0]
dis["kw"] = dis[TOPIC_COLS].apply(lambda r: sorted(t for t, c in zip(TOPICS, TOPIC_COLS) if r[c]), axis=1)
dis = dis[dis["kw"].apply(set) != dis["jev_topics"].apply(set)]
print(f"Disagree on at least one topic: {len(dis) / has.sum():.0%} of applications\\n")
for r in dis.sample(6, random_state=1).itertuples():
    print(textwrap.shorten(r.description, 260))
    print(f"   keyword: {r.kw}\\n   JEV:     {r.jev_topics}\\n")
""")

md("## 4 · Counting conditions")
code("""
both = d[has & d["jev_n_conditions"].notna()]
print(f"JEV and the regex count agree on {(both['jev_n_conditions'] == both['n_cond']).mean():.0%} of applications")
print(f"Regex says 1 but JEV says more: {((both['n_cond'] == 1) & (both['jev_n_conditions'] > 1)).mean():.1%}")
print(f"Topic named in the request (JEV): {d.loc[has, 'jev_topic_stated'].mean():.0%}")
print(f"Request says the details are needed before work starts: {(d.loc[has, 'jev_pre_commencement'] == True).mean():.0%}")
pd.crosstab(both["n_cond"].clip(upper=7), both["jev_n_conditions"].clip(upper=7), margins=True,
            rownames=["regex count"], colnames=["JEV count"])
""")

md("""
## 5 · Do the page's findings survive?

Recalculate the page's two topic-based findings using JEV's labels. Late = decided after 8 weeks, or still undecided past 8 weeks. Refused = refused or only partly approved (decided applications only).
""")
code("""
dd = d[has].copy()
dd["refused"] = dd["outcome"].isin(["refused", "partial"]).astype(float).where(dd["decided"] == 1)

def topic_table(flag):
    out = []
    for t in TOPICS:
        m = flag(t)
        if m.sum() >= 150:
            out.append({"topic": t, "applications": int(m.sum()), "late %": 100 * dd.loc[m, "late"].mean(),
                        "refused or partial %": 100 * dd.loc[m, "refused"].mean()})
    return pd.DataFrame(out).set_index("topic")

kw_tab = topic_table(lambda t: dd[slug(t)].astype(bool))
jev_tab = topic_table(lambda t: dd["jev_topics"].apply(lambda ts: t in ts))
both_tab = jev_tab.join(kw_tab, lsuffix=" (JEV)", rsuffix=" (keyword)", how="outer")
print(f"All applications: late {100 * dd['late'].mean():.0f}%, refused or partial {100 * dd['refused'].mean():.0f}%")
both_tab.sort_values("late % (JEV)", ascending=False).round(0)
""")
code("""
dd["n_jev"] = dd["jev_n_conditions"].fillna(dd["n_cond"])
bund = pd.DataFrame({
    "applications (JEV count)": dd.groupby(pd.cut(dd["n_jev"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]))["late"].size(),
    "late % (JEV count)": dd.groupby(pd.cut(dd["n_jev"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]))["late"].mean() * 100,
    "late % (regex count)": dd.groupby(pd.cut(dd["n_cond"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]))["late"].mean() * 100,
    "refused % (JEV count)": dd.groupby(pd.cut(dd["n_jev"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]))["refused"].mean() * 100,
})
bund.round(0)
""")

md("""
## 6 · Does the late model improve with JEV features?

Swap the 16 keyword topic flags and the regex condition count for JEV's versions, keeping everything else the same. Each version is fitted with 5 random seeds on the same time split.
""")
code("""
jt = pd.DataFrame({"jev_" + slug(t)[2:]: d["jev_topics"].apply(lambda ts: int(isinstance(ts, list) and t in ts)) for t in TOPICS})
d2 = pd.concat([d, jt], axis=1)
d2["jev_n_cond_c"] = d2["jev_n_conditions"].fillna(d2["n_cond"]).clip(upper=10)
d2["jev_n_topics"] = jt.sum(axis=1)
d2["jev_topic_stated"] = d2["jev_topic_stated"].fillna(False).astype(int)
d2["jev_precommencement"] = d2["jev_pre_commencement"].map({True: 1, False: 0})
JEV_COLS = list(jt.columns) + ["jev_n_cond_c", "jev_n_topics", "jev_topic_stated", "jev_precommencement"]
BASE_NO_TEXT = [c for c in NUM if c not in TOPIC_COLS + ["n_cond_c", "n_topics", "any_precommencement"]]

train, test = time_split(d2)
cat_levels = {c: train[c].astype("category").cat.categories for c in CAT}
GBM = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def prep(X, feats):
    X = X[feats].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X

rows = []
for label, num in [("No description features", BASE_NO_TEXT),
                   ("Keyword features (current)", NUM),
                   ("JEV features instead", BASE_NO_TEXT + JEV_COLS),
                   ("Both", NUM + JEV_COLS)]:
    aucs = []
    for seed in range(5):
        m = lgb.LGBMClassifier(**GBM, random_state=seed).fit(prep(train, CAT + num), train["late"])
        aucs.append(roc_auc_score(test["late"], m.predict_proba(prep(test, CAT + num))[:, 1]))
    rows.append({"features": label, "Late AUC": np.mean(aucs), "± (5 seeds)": np.std(aucs)})
pd.DataFrame(rows).set_index("features").round(3)
""")

md("## 7 · Summary")
code("""
print("Filled in from the results above when the notebook runs:")
top_false = cmp_topics.sort_values("keyword tags JEV agrees with %").head(3)
print("Least reliable keyword topics:", ", ".join(f"{t} ({v:.0f}% confirmed)" for t, v in top_false["keyword tags JEV agrees with %"].items()))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
path = NOTEBOOKS / "04_description_topics.ipynb"
nbf.write(nb, path)
print("wrote", path)
