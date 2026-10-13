"""
Decision-support sweep (not part of `main.py train`).

Runs the scikit-learn model under every combination of the framing choices
in train.CONFIG so you can see what each decision does to the results
before you commit to one. Writes artifacts/experiments.csv.

    uv run python experiments.py
"""

import copy
import itertools

import numpy as np
import pandas as pd

import train

rows = []
for stage, split, max_year, neg in itertools.product(
        train.STAGES, ["random", "time"], [None, 2009, 2008],
        ["drop", "clip", "flag_and_clip"]):
    cfg = copy.deepcopy(train.CONFIG)
    cfg.update(meeting_stage=stage, split=split, max_founded_year=max_year,
               negative_age=neg)
    out = train.run(cfg, torch_models=False, verbose=False)
    r, b = out["results"]["sklearn"], out["results"]["baseline"]
    test_years = out["df"]["founded_at"].dt.year.iloc[out["split"]["test"]]
    rows.append({
        "stage": stage, "split": split, "max_year": max_year, "neg_age": neg,
        "n_train": out["n"]["train"], "n_test": out["n"]["test"],
        "test_years": f"{test_years.min()}-{test_years.max()}",
        "test_acq_rate": round(out["y"][out["split"]["test"]].mean(), 3),
        "C": out["C"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
        "log_loss": r["log_loss"], "baseline_log_loss": b["log_loss"],
        "cost_per_100": r["cost_per_100"], "baseline_cost_per_100": b["cost_per_100"],
    })

res = pd.DataFrame(rows)
train.ART.mkdir(exist_ok=True)
res.to_csv(train.ART / "experiments.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))

# Cost ratio: at which missed-winner/wasted-meeting ratio does the
# validation-optimal policy collapse to "meet everyone"?
print("\nThreshold chosen on validation vs cost ratio (missed winner : wasted meeting)")
for stage in train.STAGES:
    cfg = copy.deepcopy(train.CONFIG)
    cfg["meeting_stage"] = stage
    out = train.run(cfg, torch_models=False, verbose=False)
    y_va = out["y"][out["split"]["val"]]
    p_va = out["probs"]["sklearn"]["val"]
    line = []
    for ratio in [0.5, 1, 1.5, 2, 3, 5, 10]:
        c = dict(cfg, cost_wasted_meeting=1.0, cost_missed_winner=ratio)
        line.append(f"{ratio}:1 -> {train.best_threshold(y_va, p_va, c):.2f}")
    print(f"  {stage:17s}", " | ".join(line))
