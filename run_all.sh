#!/usr/bin/env bash
set -euo pipefail
python scripts/backtest.py
python scripts/eda.py
python scripts/train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python scripts/make_report.py
