"""Rolling-origin backtest that mimics the real task (train on the past,
predict the next two months), plus ablations and a cold-city check.

Usage:  python scripts/backtest.py
Writes: reports/backtest_results.csv, reports/cold_city_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.model import FreightRateModel  # noqa: E402
from freight.pipeline import metrics, prepare  # noqa: E402

FOLDS = [  # (train end, test end): train on the past, predict the next 2 months
    # Same shape as the real task (Oct observed -> predict Nov-Dec): the first
    # month of a quarter is in training, the 2nd and 3rd months are predicted.
    ("2025-05-01", "2025-07-01"),  # Apr known -> May-Jun
    ("2025-08-01", "2025-10-01"),  # Jul known -> Aug-Sep
    # Stress test: crosses a quarter boundary (Oct level never observed).
    ("2025-09-01", "2025-11-01"),
]


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median rate-per-mile by equipment x distance band (simple rate-card baseline)."""
    bins = np.quantile(train["distance"], np.linspace(0, 1, 11))
    bins[0], bins[-1] = 0, np.inf
    tr = train.assign(band=pd.cut(train["distance"], bins), adj=train["posted_rate"] / train["distance"])
    table = tr.groupby(["equipment", "band"], observed=True)["adj"].median()
    te = test.assign(band=pd.cut(test["distance"], bins))
    rpm = te.set_index(["equipment", "band"]).index.map(table).to_numpy(dtype=float)
    return rpm * te["distance"].to_numpy()


def main() -> None:
    train, _, _, _ = prepare()

    # Flag likely mis-keyed labels once on all data (used ONLY to report a
    # "clean" metric; every model is still trained on its own fold only).
    flagger = FreightRateModel()
    bad = flagger.flag_outliers(train.reset_index(drop=True))
    train = train.reset_index(drop=True).assign(label_outlier=bad)
    print(f"label outliers flagged: {bad.sum()} ({bad.mean():.2%})")

    variants = {
        "baseline_rpm_table": None,
        "gbm_market_only_no_calendar": dict(use_time_effect=False),
        "final_but_keep_outliers": dict(use_time_effect=True, drop=False),
        "final (gbm + daily effect)": dict(use_time_effect=True),
    }
    rows = []
    for tr_end, te_end in FOLDS:
        tr = train[train["date"] < tr_end]
        te = train[(train["date"] >= tr_end) & (train["date"] < te_end)]
        for name, cfg in variants.items():
            if cfg is None:
                p = baseline_predict(tr, te)
            else:
                cfg = dict(cfg)
                drop = cfg.pop("drop", True)
                p = FreightRateModel(**cfg).fit(tr, drop_outliers=drop).predict(te)
            y = te["posted_rate"].to_numpy()
            clean = ~te["label_outlier"].to_numpy()
            for subset, mask in [("all_rows", np.ones_like(clean)), ("clean_rows", clean)]:
                rows.append({"fold_test": f"{tr_end[:7]}..{te_end[:7]}", "model": name, "subset": subset, **metrics(y[mask], p[mask])})
            print(rows[-1])
    res = pd.DataFrame(rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "backtest_results.csv", index=False)
    summary = res.groupby(["subset", "model"])[["MAE", "RMSE", "MAPE_%", "MedAPE_%"]].mean().round(2)
    print(summary.to_string())
    summary.to_csv(out / "backtest_summary.csv")

    # Cold-city check: hide 8 cities from training entirely (validation has 8 unseen cities).
    rng = np.random.default_rng(0)
    cities = sorted(set(train["pickup"]))
    cold = set(rng.choice(cities, 8, replace=False))
    tr_end, te_end = FOLDS[-1]
    has_cold = train["pickup"].isin(cold) | train["delivery"].isin(cold)
    tr = train[(train["date"] < tr_end) & ~has_cold]
    te = train[(train["date"] >= tr_end) & (train["date"] < te_end) & ~train["label_outlier"]]
    m = FreightRateModel().fit(tr)
    cc = []
    for subset, mask in [("seen_cities", ~te["pickup"].isin(cold) & ~te["delivery"].isin(cold)),
                         ("unseen_cities", te["pickup"].isin(cold) | te["delivery"].isin(cold))]:
        sub = te[mask]
        cc.append({"subset": subset, **metrics(sub["posted_rate"].to_numpy(), m.predict(sub))})
    cc = pd.DataFrame(cc)
    print("cold cities:", sorted(cold))
    print(cc.round(2).to_string())
    cc.to_csv(out / "cold_city_results.csv", index=False)


if __name__ == "__main__":
    main()
