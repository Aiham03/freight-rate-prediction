"""Loading, cleaning and market-context tables.

Data-quality issues handled here (see README / report for the evidence):
  * negative weights      -> sign error, use the absolute value
  * missing weights       -> left as NaN (LightGBM handles it) + a missing flag
  * weight == 47,500      -> a clip/cap, kept as-is (it is a real value at the cap)
  * missing market_index  -> filled with that day's market-wide mean
  * mis-keyed posted_rate -> detected later with out-of-fold residuals (model.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CITY_COLS = ["pickup", "delivery"]


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    return train, valid


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CITY_COLS + ["equipment"]:
        out[col] = out[col].astype(str).str.strip()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    return out


def city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city. Coordinates are constant per city in the data."""
    parts = []
    for f in frames:
        for side in CITY_COLS:
            if f"{side}_lat" in f:
                parts.append(
                    f[[side, f"{side}_lat", f"{side}_lon"]]
                    .set_axis(["city", "lat", "lon"], axis=1)
                )
    return pd.concat(parts).groupby("city")[["lat", "lon"]].median()


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Market-wide daily context built only from *feature* columns.

    market_index is essentially a daily series (within-day std ~0.025 vs
    ~0.17 across days); quote_signal has a smaller daily component. Both are
    observed for every date in train and validation, so using them is not
    target leakage. Rolling features capture the weekly cycle and the level.
    """
    a = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    d = (
        a.groupby("date")
        .agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "median"))
        .sort_index()
    )
    d = d.asfreq("D").interpolate()
    d["lmi_day"] = np.log(d["mi_day"])
    d["lmi_roll7"] = d["lmi_day"].rolling(7, min_periods=1).mean()
    d["lmi_roll28"] = d["lmi_day"].rolling(28, min_periods=1).mean()
    d["lmi_lag7"] = d["lmi_day"].shift(7).bfill()
    return d


def attach_market(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(daily, left_on="date", right_index=True, how="left")
    if "market_index" not in out:
        out["market_index"] = np.nan
    if "quote_signal" not in out:
        out["quote_signal"] = np.nan
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["mi_day"])
    out["quote_signal"] = out["quote_signal"].fillna(out["qs_day"])
    return out


def attach_coords(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in CITY_COLS:
        if f"{side}_lat" not in out:
            out[f"{side}_lat"] = out[side].map(cities["lat"])
            out[f"{side}_lon"] = out[side].map(cities["lon"])
    return out
