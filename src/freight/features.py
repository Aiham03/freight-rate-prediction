"""Feature engineering.

Design choices:
  * Cities enter only through coordinates (not IDs). 8 validation cities never
    appear in training (12% of validation rows), and after conditioning on
    coordinates the per-city residual effect is ~0.2%, so IDs add risk, not signal.
  * quote_signal enters only as the deviation from its daily median: the raw
    value carries a day-level component that let the trees soak up the
    calendar trend (found in backtest diagnostics).
  * No calendar or day-level market features go into the tree model: the
    validation period (Nov-Dec) is outside the training range, trees cannot
    extrapolate, and day-level market features are confounded with the
    calendar drift. Those effects are modelled by DailyEffect in model.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

FEATURES = [
    "log_distance",
    "weight",
    "weight_missing",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "d_lat",
    "d_lon",
    "bearing",
    "lmi_load_dev",
    "qs_load_dev",
]

# +1 = prediction may only increase with the feature (log-rate target)
MONOTONE = {"log_distance": 1, "weight": 1}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_distance"] = np.log(df["distance"].clip(lower=1))
    X["weight"] = df["weight"]
    X["weight_missing"] = df["weight"].isna().astype(int)
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    X["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    X["bearing"] = np.degrees(np.arctan2(X["d_lat"], X["d_lon"]))
    # load-level deviations only; the day-level market enters via DailyEffect
    X["lmi_load_dev"] = np.log(df["market_index"]) - df["lmi_day"]
    X["qs_load_dev"] = df["quote_signal"] - df["qs_day"]
    return X[FEATURES]


def monotone_vector() -> list[int]:
    return [MONOTONE.get(f, 0) for f in FEATURES]
