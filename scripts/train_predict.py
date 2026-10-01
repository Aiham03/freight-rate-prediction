"""Train the final model on ALL of train_test.csv and write both deliverables.

Usage:  python scripts/train_predict.py
Writes: validation_predictions.csv
        data/december_chart_inputs.csv          (predicted_rate filled in place)
        reports/feature_importance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.features import FEATURES  # noqa: E402
from freight.model import FreightRateModel  # noqa: E402
from freight.pipeline import prepare, prepare_december  # noqa: E402

DATA = ROOT / "data"


def main() -> None:
    train, valid, daily, cities = prepare()
    model = FreightRateModel().fit(train)
    de = model.daily_
    print(f"dropped {model.outlier_mask_.sum()} label outliers; beta={de.beta:.3f} ramp={np.round(de.gamma, 4)} "
          f"levels={ {k: round(float(v), 4) for k, v in de.levels.items()} }")

    # 1) validation predictions, in template order
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(valid), index=valid["load_id"].values)
    template["predicted_rate"] = template["load_id"].map(pred).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template[["load_id", "predicted_rate"]].to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"wrote validation_predictions.csv ({len(template):,} rows)")

    # 2) December chart, filled in place (original 7 columns, original order)
    dec_path = DATA / "december_chart_inputs.csv"
    dec_out = pd.read_csv(dec_path)
    dec_feat = prepare_december(dec_path, daily, cities)
    dec_out["predicted_rate"] = np.round(model.predict(dec_feat), 2)
    dec_out.to_csv(dec_path, index=False)
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))

    imp = pd.Series(model.gbm_.booster_.feature_importance("gain"), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(ROOT / "reports" / "feature_importance.csv", header=["gain_share"])


if __name__ == "__main__":
    main()
