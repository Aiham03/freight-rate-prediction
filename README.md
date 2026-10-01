# Freight Rate Prediction

Predicts `posted_rate` for 12,000 Nov–Dec 2025 loads from 48,000 labelled Jan–Oct 2025 loads.
The validation report is `reports/Freight_Rate_Report.pdf`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python scripts/train_predict.py   # fit on all training data -> validation_predictions.csv + fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Optional steps to reproduce the analysis. They take about 10 minutes on one CPU core.

```bash
python scripts/backtest.py        # rolling-origin + cold-city backtests -> reports/backtest_*.csv
python scripts/eda.py             # data-quality audit + figures -> reports/eda_summary.txt, reports/figures/
python scripts/make_report.py     # -> reports/Freight_Rate_Report.pdf
```

Or run everything with `bash run_all.sh`.

## Layout

```
data/                     provided inputs (train_test, validation, template, december_chart_inputs)
src/freight/data.py       loading, cleaning, daily market table, city coordinate lookup
src/freight/features.py   load-level features for the tree model
src/freight/model.py      FreightRateModel = LightGBM + DailyEffect, outlier filter, backfitting
src/freight/pipeline.py   shared preparation + metrics
scripts/                  backtest, eda, train_predict, make_report
score.py                  provided scorer (unchanged)
validation_predictions.csv, scorer_results/candidate_december.png   deliverables
```

## Approach in brief

- **Cleaning.** Mis-keyed rates (1.4%) are clustered at about 0.29x and 3.4x the expected rate. They are flagged by out-of-fold residuals and dropped from training. Negative weights are treated as sign errors and replaced by their absolute value. Missing `market_index` is filled with the daily mean. Missing weight is left to LightGBM.
- **Model.** `log(rate) = LightGBM(distance, coordinates, equipment, weight, load-level signal deviations) + DailyEffect(date)`.
  - `DailyEffect = beta * 7-day mean log(market_index) + quarter level + convex within-quarter ramp`.
  - The two parts are fitted by backfitting, and a smearing correction is applied when converting back from log.
  - Day-level effects are kept out of the trees because trees cannot extrapolate to Nov–Dec. When trees were given day-level market features, they used them as a proxy for the calendar.
- **Validation.** Rolling-origin time splits with a 2-month horizon, built to match the real task (first month of the quarter known, next two months predicted), plus a stress fold across a quarter boundary and a cold-city holdout. Mean backtest MAPE on clean labels is about 2.0%, versus 4.5% for a rate-card baseline.

## Use of AI tools

I wrote the code and the modelling approach. I used Claude Code (Anthropic) for pre-submission testing and repo setup:

- Ran `score.py` on both deliverables, and ran the full pipeline (`run_all.sh`) from scratch in a clean environment to confirm the outputs reproduce exactly.
- Reviewed the code for bugs and target leakage, and cross-checked the numbers in the report against the CSV outputs.
- Suggested two wording changes in `scripts/make_report.py`: a more precise cold-city comparison and a sentence stating the main December assumption. The report was regenerated afterwards.
- Added `.gitattributes` so the PDF and `run_all.sh` keep the correct line endings.
