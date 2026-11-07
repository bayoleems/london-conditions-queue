"""Label discharge-of-condition descriptions with Jev (~typesafe/jev-latest) via OpenRouter's decisions API.

Jev answers narrow, typed questions about a piece of state. Each description is one request:
  - one yes/no ("noul", a probability 0-1) question per condition topic, about the conditions being
    discharged only, not the copied text of the original permission
  - how many conditions are being discharged ("choice")
  - whether the request says what the conditions cover ("noul")
  - whether it says the details are needed before work starts ("choice")

Output: outputs/jev_topics.jsonl, one JSON object per application (cached; reruns skip what is done):
  {"name", "topics": [...p >= 0.5], "topic_probs": {topic: p}, "n_conditions": int|null,
   "topic_stated": bool, "pre_commencement": bool|null, "model": "..."}

Key: the OPENROUTER_API_KEY environment variable, or a .env file at the repo root (never committed).

Run:
  python -m conditions_queue.classify_jev --sample 100   # test on a sample
  python -m conditions_queue.classify_jev                # everything not yet labelled
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from conditions_queue.paths import DISCHARGE_CSV, ENV_FILE, JEV_JSONL, PARQUET

OUT = JEV_JSONL
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"

CONTEXT = ("A UK planning application to discharge (sign off) planning conditions. The description first says which "
           "condition(s) are being discharged, often with a short name in brackets. It then usually copies the ORIGINAL "
           "permission's description (e.g. 'for erection of 41 flats with cycle storage and landscaping'). Judge only the "
           "conditions being discharged and ignore the copied permission text.")

# Same 16 topics and definitions as the keyword tagger in analysis.py
TOPIC_DEFS = {
    "Accessibility": "accessible or wheelchair housing (M4(2), M4(3)), accessibility",
    "Archaeology & heritage": "archaeology, written scheme of investigation, heritage recording",
    "Boundary & lighting": "boundary treatment, fences, walls, external lighting",
    "Construction logistics / management": "construction logistics or management plans (CLP, CMP, CEMP), demolition or construction method statements",
    "Contamination & ground": "land contamination, remediation, site investigation, verification reports",
    "Drainage & flood": "SuDS, surface or foul water drainage, flood risk, sewage disposal",
    "Ecology & biodiversity": "ecology, biodiversity, bats, birds, bird or bat boxes, green or brown roofs, habitats",
    "Energy & carbon": "energy strategy, SAP, BREEAM, carbon, overheating, PV, heat pumps, water efficiency or usage",
    "Fire safety": "fire safety, fire strategy",
    "Highways & travel": "highway works, s278, travel plans, delivery and servicing plans, access",
    "Landscaping & trees": "hard or soft landscaping, planting, tree protection, arboricultural statements",
    "Materials & design detail": "materials, samples, brick, cladding, windows, balustrades, detailed design drawings",
    "Noise, vibration & air": "noise, sound insulation, vibration, air quality, dust, ventilation, odour",
    "Piling & structure": "piling, foundations, basements, structural details, site levels",
    "Secured by Design": "Secured by Design, crime prevention",
    "Waste, cycle & parking": "refuse or recycling stores, cycle stores, car parking, EV charging, car clubs",
}
TOPIC_KEYS = {t: "t" + str(i) for i, t in enumerate(TOPIC_DEFS)}
COUNT_CHOICES = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                 "8": "Eight", "9": "Nine", "10+": "Ten or more", "unclear": "Cannot tell from the text"}

QUESTIONS = {key: {"type": "noul",
                   "instructions": f"Do any of the conditions being discharged cover {defn}?",
                   "criteria": {"true": "At least one condition being discharged covers this",
                                "false": "None of the conditions being discharged covers this (mentions in the copied permission text don't count)"}}
             for (t, defn), key in zip(TOPIC_DEFS.items(), TOPIC_KEYS.values())}
QUESTIONS["n_conditions"] = {"type": "choice", "instructions": "How many separate conditions are being discharged? Count '15A and 15B' as two.",
                             "criteria": COUNT_CHOICES}
QUESTIONS["topic_stated"] = {"type": "noul", "instructions": "Does the request itself say what the condition(s) cover, rather than only giving condition numbers?",
                             "criteria": {"true": "It names or describes what at least one condition covers", "false": "It only gives condition numbers"}}
QUESTIONS["pre_commencement"] = {"type": "choice", "instructions": "Does the request say the details are required before work starts?",
                                 "criteria": {"yes": "Says pre-commencement, prior to commencement or before works start",
                                              "no": "Says they are needed at a later stage (e.g. prior to occupation)",
                                              "not_stated": "Doesn't say"}}


def api_key():
    """OpenRouter key from the environment, or from OPENROUTER_API_KEY=... in the repo's .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY, or put OPENROUTER_API_KEY=... in a .env file at the repo root.")
    return key


def load_items():
    """Every discharge application's name and description (first 1,500 characters)."""
    apps = pd.read_csv(DISCHARGE_CSV, usecols=["name"])
    desc = pd.read_parquet(PARQUET, columns=["name", "description"])
    df = apps.merge(desc, on="name")
    df["description"] = df["description"].fillna("").str.slice(0, 1500)
    return df


def to_row(name, answers, model):
    probs = {t: round(float(answers[k]["noul"]), 3) for t, k in TOPIC_KEYS.items() if k in answers}
    n = answers.get("n_conditions", {}).get("choice")
    n_val = None if n in (None, "unclear") else (10 if n == "10+" else int(n))
    pre = answers.get("pre_commencement", {}).get("choice")
    return {"name": name, "topics": sorted(t for t, p in probs.items() if p >= 0.5), "topic_probs": probs,
            "n_conditions": n_val, "topic_stated": answers.get("topic_stated", {}).get("noul", 0) >= 0.5,
            "pre_commencement": {"yes": True, "no": False}.get(pre), "model": model}


def classify_one(session, key, name, description):
    body = {"model": MODEL, "state": [CONTEXT, "Description: " + description], "questions": QUESTIONS}
    err = None
    for attempt in range(4):
        try:
            r = session.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                           "X-OpenRouter-Title": "House London conditions"},
                             data=json.dumps(body), timeout=(15, 90))
            if r.status_code == 402:
                raise SystemExit("OpenRouter says the account is out of credits (402). Stopping.")
            r.raise_for_status()
            j = r.json()
            missing = [k for k in QUESTIONS if k not in j.get("answers", {})]
            if missing:
                raise ValueError(f"answers missing for {missing[:3]}")
            return to_row(name, j["answers"], j.get("model")), (j.get("usage") or {}).get("cost") or 0.0
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - retry any API/parse failure
            err = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"{name}: failed after retries: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="label a random sample of this many (for testing)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=4.0, help="stop once this many dollars have been spent")
    args = ap.parse_args()

    df = load_items()
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {json.loads(line)["name"] for line in f}
    todo = df[~df["name"].isin(done)]
    if args.sample:
        todo = todo.sample(min(args.sample, len(todo)), random_state=len(done))
    print(f"{len(df):,} descriptions · {len(done):,} already done · {len(todo):,} to label with {MODEL}", flush=True)

    key, session = api_key(), requests.Session()
    stop = threading.Event()

    def work(item):
        return None if stop.is_set() else classify_one(session, key, item.name, item.description)

    total, n_done, failed, t0 = 0.0, 0, 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool, open(OUT, "a") as f:
        futures = [pool.submit(work, item) for item in todo.itertuples()]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except SystemExit as e:  # out of credits
                print(e, flush=True)
                stop.set()
                continue
            except RuntimeError as e:
                failed += 1
                print("  skipped:", e, flush=True)
                continue
            if result is None:
                continue
            row, cost = result
            f.write(json.dumps(row) + "\n")
            f.flush()
            total += cost
            n_done += 1
            if n_done % 250 == 0 or n_done == len(todo):
                print(f"  {n_done:,}/{len(todo):,} · ${total:.3f} · {n_done / max(time.time() - t0, 1):.1f}/s", flush=True)
            if total >= args.max_cost and not stop.is_set():
                print(f"Reached the ${args.max_cost:.2f} spending cap. Stopping.", flush=True)
                stop.set()
    print(f"Done: {n_done:,} labelled, {failed} failed, ${total:.3f} "
          f"(${1000 * total / max(n_done, 1):.3f} per 1,000)", flush=True)


if __name__ == "__main__":
    main()
