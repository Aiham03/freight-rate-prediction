"""End-to-end preparation shared by backtest, training and prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D


def prepare(data_dir: Path = D.DATA_DIR):
    train_raw, valid_raw = D.load_raw(data_dir)
    train, valid = D.clean(train_raw), D.clean(valid_raw)
    daily = D.daily_market(train, valid)
    cities = D.city_lookup(train, valid)
    train = D.attach_market(train, daily)
    valid = D.attach_market(valid, daily)
    return train, valid, daily, cities


def prepare_december(path: Path, daily: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """December chart rows carry only lane/equipment/weight/date.

    Coordinates come from the city lookup; market_index / quote_signal are
    unknown for a hypothetical load, so they are filled with that day's
    market-wide value observed in validation.csv features.
    """
    dec = pd.read_csv(path, parse_dates=["date"])
    df = D.clean(dec.drop(columns=["predicted_rate"]))
    df = D.attach_coords(df, cities)
    return D.attach_market(df, daily)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y) * 100),
        "n": int(len(y)),
    }
