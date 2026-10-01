"""Exploratory analysis + data-quality audit. Produces figures in reports/figures
and a text summary in reports/eda_summary.txt.

Usage:  python scripts/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight import data as D  # noqa: E402
from freight.features import build_features  # noqa: E402
from freight.model import FreightRateModel  # noqa: E402
from freight.pipeline import prepare  # noqa: E402

FIG = ROOT / "reports" / "figures"
TEAL = "#064A56"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tr_raw, va_raw = D.load_raw()
    lines = []
    say = lambda s: (print(s), lines.append(s))  # noqa: E731

    say(f"train rows {len(tr_raw):,} | {tr_raw.date.min():%Y-%m-%d}..{tr_raw.date.max():%Y-%m-%d}")
    say(f"valid rows {len(va_raw):,} | {va_raw.date.min():%Y-%m-%d}..{va_raw.date.max():%Y-%m-%d}")
    say(f"duplicate load_id / duplicate feature rows: {tr_raw.load_id.duplicated().sum()} / "
        f"{tr_raw.drop(columns='load_id').duplicated().sum()}")
    for name, f in [("train", tr_raw), ("valid", va_raw)]:
        say(f"[{name}] missing weight {f.weight.isna().sum()}, missing market_index {f.market_index.isna().sum()}, "
            f"negative weight {(f.weight < 0).sum()}, weight at 47,500 cap {(f.weight.abs() == 47500).sum()}")
    tc = set(tr_raw.pickup) | set(tr_raw.delivery)
    unseen = sorted((set(va_raw.pickup) | set(va_raw.delivery)) - tc)
    n_unseen = (~va_raw.pickup.isin(tc) | ~va_raw.delivery.isin(tc)).sum()
    say(f"cities only in validation: {unseen} -> {n_unseen} validation rows ({n_unseen / len(va_raw):.1%})")
    say(f"market_index mean train {tr_raw.market_index.mean():.3f} vs valid {va_raw.market_index.mean():.3f}")
    g = tr_raw.groupby("date").market_index
    say(f"market_index within-day std {g.std().mean():.3f} vs across-day std {g.mean().std():.3f}")

    train, valid, daily, _ = prepare()
    train = train.reset_index(drop=True)

    # --- label outliers -----------------------------------------------------
    m = FreightRateModel(use_time_effect=True)
    bad = m.flag_outliers(train)
    ratio = np.exp(m.oof_resid_)
    say(f"label outliers (|log resid|>0.35): {bad.sum()} ({bad.mean():.2%}); "
        f"low {(ratio < 0.7).sum()}, high {(ratio > 1.42).sum()}; "
        f"median ratio low {np.median(ratio[ratio < 0.7]):.2f}, high {np.median(ratio[ratio > 1.42]):.2f}")
    say(f"residual std on clean rows: {m.oof_resid_[~bad].std():.4f} (log units)")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(np.log10(ratio), bins=120, color=TEAL)
    ax.set_yscale("log")
    ax.axvline(np.log10(np.exp(0.35)), color="crimson", ls="--")
    ax.axvline(np.log10(np.exp(-0.35)), color="crimson", ls="--")
    ax.set_xlabel("log10(actual / out-of-fold predicted rate)")
    ax.set_ylabel("loads (log scale)")
    ax.set_title("Label outliers: two clusters of mis-keyed rates (~1.4%)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "label_outliers.png", dpi=150); plt.close(fig)

    # --- rate per mile vs distance -----------------------------------------
    c = train[~bad]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for eq, col in zip(["Dry Van", "Reefer", "Flatbed"], [TEAL, "#2A9D8F", "#E9C46A"]):
        s = c[c.equipment == eq].sample(1500, random_state=0)
        ax.scatter(s.distance, s.posted_rate / s.distance, s=4, alpha=0.4, label=eq, color=col)
    ax.set_xlabel("distance (mi)"); ax.set_ylabel("rate per mile ($)")
    ax.set_ylim(1.2, 4); ax.legend(); ax.set_title("Rate per mile falls with distance; Reefer > Flatbed > Dry Van", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "rpm_distance.png", dpi=150); plt.close(fig)

    # --- time structure after controlling for load features ----------------
    tm = FreightRateModel().fit(c, drop_outliers=False)
    y = np.log(c.posted_rate.to_numpy())
    day_resid = pd.Series(y - tm.gbm_.predict(build_features(c))).groupby(c.date.values).mean()
    beta = tm.daily_.beta
    days = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-12-31")})
    days = days.merge(daily[["lmi_roll7"]], left_on="date", right_index=True)
    calendar_only = tm.daily_.predict(days) - beta * days.lmi_roll7.to_numpy()
    lmi7 = daily.lmi_roll7.reindex(day_resid.index).to_numpy()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(daily.index, daily.lmi_day, color="#9DAFB3", lw=1, label="daily")
    ax[0].plot(daily.index, daily.lmi_roll7, color=TEAL, lw=2, label="7-day mean (used)")
    ax[0].axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color="grey", alpha=0.15)
    ax[0].set_title("log market_index: strong weekly cycle; grey = validation period", loc="left")
    ax[0].legend(fontsize=8)
    ax[1].plot(day_resid.index, day_resid.values - beta * lmi7, color=TEAL, lw=1,
               label="daily mean residual after load + market effects")
    ax[1].plot(days.date, calendar_only, color="crimson", lw=2, label="fitted quarter level + ramp (extrapolated Nov-Dec)")
    ax[1].axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color="grey", alpha=0.15)
    ax[1].set_title("Calendar effect: rates ramp through each quarter and reset on the 1st", loc="left")
    ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "time_structure.png", dpi=150); plt.close(fig)
    say(f"daily effect: beta(lmi_roll7)={beta:.3f}, ramp={np.round(tm.daily_.gamma, 4)}, "
        f"levels={ {k: round(float(v), 4) for k, v in tm.daily_.levels.items()} }")

    # --- weight effect -----------------------------------------------------
    wb = pd.cut(c.weight, [0, 15000, 20000, 25000, 30000, 35000, 40000, 47499, 47501])
    rr = pd.Series(np.log(c.posted_rate.to_numpy()) - tm.predict_log(c)) + np.log(tm.smear_)
    say("weight band mean log-residual (should be ~0 after model): "
        + ", ".join(f"{k}: {v:+.3f}" for k, v in rr.groupby(wb.values, observed=True).mean().items()))

    (ROOT / "reports" / "eda_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
