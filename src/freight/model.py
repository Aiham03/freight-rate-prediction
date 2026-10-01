"""FreightRateModel = LightGBM on load-level features + a linear daily market effect.

log(rate) = GBM(lane geometry, equipment, weight, load-level signals) + DailyEffect(date)

Why the split: in a first version the tree model also saw daily market
features (same-day / rolling market_index). Because the market index and the
calendar drift are confounded inside Jan-Aug, the trees used the market
index as a proxy for time and the backtest error doubled on Sep-Oct. Moving
everything that is constant within a day into a small, interpretable linear
model (smoothed market index + quarter level + within-quarter ramp) removes
that confounding and lets the trees focus on lane/equipment/weight.
The two parts are estimated by backfitting (alternate fits on each other's
residuals).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .features import build_features, monotone_vector

ORIGIN = pd.Timestamp("2025-01-01")

GBM_PARAMS = dict(
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def quarter_frac(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    q = dates.dt.to_period("Q")
    start, end = q.dt.start_time, q.dt.end_time.dt.normalize()
    return ((dates - start).dt.days / ((end - start).dt.days + 1)).to_numpy()


def quarter_key(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).dt.to_period("Q").astype(str).to_numpy()


@dataclass
class DailyEffect:
    """Market-wide daily component of log(rate), fitted by weighted least squares
    on daily mean residuals:

        effect(day) = beta * lmi_roll7(day) + level[quarter] + g1 * f + g2 * f^2,
        f = fraction of the quarter elapsed

    * lmi_roll7: 7-day mean of log market_index. Rates respond to the smoothed
      market, not the day-to-day weekly oscillation (it fits far better than
      the same-day value).
    * level[quarter]: each quarter has its own base level (they step up through
      the year). For Nov-Dec the Q4 level is identified from October.
    * g1*f + g2*f^2: rates ramp up (convexly) through every quarter and reset at the
      start of the next one (quarter-end capacity squeeze).
    A quarter never seen in training gets last level + average step (only
    relevant for backtests that cross a quarter boundary).
    """

    use_time: bool = True
    quadratic: bool = False
    beta: float = 0.0
    gamma: float = 0.0
    levels: dict = field(default_factory=dict)

    def _design(self, dates, lmi7, quarters):
        cols = [lmi7]
        if self.use_time:
            cols += [(quarter_key(dates) == q).astype(float) for q in quarters]
            f = quarter_frac(dates)
            cols += [f, f**2] if self.quadratic else [f]
        else:
            cols += [np.ones(len(lmi7))]
        return np.column_stack(cols)

    def fit(self, df: pd.DataFrame, resid: np.ndarray) -> "DailyEffect":
        g = (
            pd.DataFrame({"date": df["date"].values, "lmi7": df["lmi_roll7"].values, "r": resid})
            .groupby("date").agg(r=("r", "mean"), lmi7=("lmi7", "first"), n=("r", "size")).reset_index()
        )
        quarters = sorted(set(quarter_key(g["date"]))) if self.use_time else []
        A = self._design(g["date"], g["lmi7"].to_numpy(), quarters)
        w = np.sqrt(g["n"].to_numpy())
        coef, *_ = np.linalg.lstsq(A * w[:, None], g["r"].to_numpy() * w, rcond=None)
        self.beta = float(coef[0])
        if self.use_time:
            k = 2 if self.quadratic else 1
            self.levels = dict(zip(quarters, coef[1:-k]))
            self.gamma = coef[-k:].copy()
        else:
            self.levels, self.gamma = {"_all": float(coef[1])}, 0.0
        return self

    def level_for(self, q: str) -> float:
        if not self.use_time:
            return self.levels["_all"]
        if q in self.levels:
            return self.levels[q]
        vals = [self.levels[k] for k in sorted(self.levels)]
        step = float(np.mean(np.diff(vals))) if len(vals) > 1 else 0.0
        n_ahead = (pd.Period(q) - pd.Period(sorted(self.levels)[-1])).n
        return vals[-1] + step * n_ahead

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        lv = np.array([self.level_for(q) for q in quarter_key(df["date"])])
        ramp = 0.0
        if self.use_time:
            f = quarter_frac(df["date"])
            F = np.column_stack([f, f**2]) if self.quadratic else f[:, None]
            ramp = F @ np.atleast_1d(self.gamma)
        return self.beta * df["lmi_roll7"].to_numpy() + lv + ramp


@dataclass
class FreightRateModel:
    use_time_effect: bool = True
    quadratic_ramp: bool = True  # convex ramp: wins on both in-quarter backtest folds
    outlier_threshold: float = 0.35  # |log residual| ~ rate off by >42%
    backfit_iters: int = 2
    gbm_params: dict = field(default_factory=lambda: dict(GBM_PARAMS))
    seed: int = 42

    def _gbm(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            **self.gbm_params, monotone_constraints=monotone_vector(), random_state=self.seed
        )

    def _backfit(self, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame):
        # Start with the daily effect on raw targets (lane mix is ~random across
        # days), so the trees never get first claim on day-level variation.
        de = DailyEffect(use_time=self.use_time_effect, quadratic=self.quadratic_ramp).fit(df, y - y.mean())
        offset = de.predict(df)
        gbm = None
        for _ in range(self.backfit_iters):
            gbm = self._gbm().fit(X, y - offset)
            de.fit(df, y - gbm.predict(X))
            offset = de.predict(df)
        return gbm, de

    def flag_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """Out-of-fold residuals from a quick model; large ones are mis-keyed rates."""
        X, y = build_features(df), np.log(df["posted_rate"].to_numpy())
        resid = np.zeros(len(df))
        quick = dict(self.gbm_params, n_estimators=300, learning_rate=0.1)
        for tr, te_idx in KFold(5, shuffle=True, random_state=self.seed).split(X):
            m = FreightRateModel(self.use_time_effect, self.quadratic_ramp, backfit_iters=2, gbm_params=quick)
            gbm, de = m._backfit(X.iloc[tr], y[tr], df.iloc[tr])
            pred = gbm.predict(X.iloc[te_idx]) + de.predict(df.iloc[te_idx])
            resid[te_idx] = y[te_idx] - pred
        self.oof_resid_ = resid
        return np.abs(resid) > self.outlier_threshold

    def fit(self, df: pd.DataFrame, drop_outliers: bool = True) -> "FreightRateModel":
        df = df.reset_index(drop=True)
        if drop_outliers:
            self.outlier_mask_ = self.flag_outliers(df)
            df = df.loc[~self.outlier_mask_].reset_index(drop=True)
        X, y = build_features(df), np.log(df["posted_rate"].to_numpy())
        self.gbm_, self.daily_ = self._backfit(X, y, df)
        # smearing correction: exp(E[log]) underestimates E[rate]
        resid = y - self.predict_log(df)
        self.smear_ = float(np.mean(np.exp(resid)))
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.gbm_.predict(build_features(df)) + self.daily_.predict(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df)) * getattr(self, "smear_", 1.0)
