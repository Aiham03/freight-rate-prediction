"""Build reports/Freight_Rate_Report.pdf from the backtest/EDA outputs.

Usage:  python scripts/make_report.py   (after backtest.py, eda.py, train_predict.py, score.py)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"
TEAL = colors.HexColor("#064A56")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=TEAL, alignment=TA_LEFT, fontSize=20, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=TEAL, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.2, textColor=colors.HexColor("#455A60"))


def table(df: pd.DataFrame, widths=None, bold_row=None) -> Table:
    HD = ParagraphStyle("HD", parent=CAP, textColor=colors.white, fontName="Helvetica-Bold")
    cell = lambda v: v if isinstance(v, Paragraph) else Paragraph(str(v), CAP)  # noqa: E731
    data = [[Paragraph(str(c), HD) for c in df.columns]] + [[cell(v) for v in row] for row in df.values.tolist()]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F4")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D6D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if bold_row is not None:
        style.append(("FONTNAME", (0, bold_row + 1), (-1, bold_row + 1), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def img(path: Path, width: float) -> Image:
    im = Image(str(path))
    im.drawHeight = width * im.imageHeight / im.imageWidth
    im.drawWidth = width
    return im


def main() -> None:
    res = pd.read_csv(REP / "backtest_results.csv")
    cold = pd.read_csv(REP / "cold_city_results.csv")
    clean = res[res.subset == "clean_rows"]
    order = ["final (gbm + daily effect)", "gbm_market_only_no_calendar", "final_but_keep_outliers", "baseline_rpm_table"]
    names = {
        "final (gbm + daily effect)": "Final: GBM + daily effect",
        "gbm_market_only_no_calendar": "Same, no calendar terms",
        "final_but_keep_outliers": "Final, outliers kept",
        "baseline_rpm_table": "Baseline rate-card",
    }
    piv = clean.pivot(index="model", columns="fold_test", values="MAPE_%").loc[order]
    piv.columns = [c.replace("..", " -> test ") for c in piv.columns]
    folds = ["Apr known: May-Jun", "Jul known: Aug-Sep", "Stress: Sep-Oct"]
    piv.columns = folds
    mae = clean.groupby("model")["MAE"].mean().loc[order]
    allrows = res[res.subset == "all_rows"].groupby("model")["MAPE_%"].mean().loc[order]
    bt = piv.map(lambda v: f"{v:.2f}").copy()
    bt.insert(0, "Model", [names[m] for m in order])
    bt["Mean MAPE %"] = [f"<b>{v:.2f}</b>" for v in piv.mean(axis=1).values]
    bt["Mean MAE $"] = [f"{v:.1f}" for v in mae.values]
    bt["MAPE % (incl. bad labels)"] = [f"{v:.2f}" for v in allrows.values]

    s = []
    s.append(Paragraph("Freight Rate Prediction - Validation &amp; Split Report", H1))
    s.append(Paragraph("Model: LightGBM on load-level features + linear daily market/calendar effect. "
                       "Target: log(posted_rate). Training data 2025-01-01..2025-10-31 (48,000 loads); "
                       "prediction set 2025-11-01..2025-12-31 (12,000 loads).", CAP))

    s.append(Paragraph("1. What the split has to reproduce", H2))
    s.append(Paragraph(
        "The validation loads are entirely <b>later in time</b> than the training loads (Nov-Dec vs Jan-Oct) and "
        "<b>12.1% of them involve one of 8 cities</b> that never appear in training. A random K-fold split would "
        "therefore over-state accuracy twice: it lets the model interpolate between neighbouring days, and it "
        "never asks it to price an unseen city. The split was designed to reproduce both conditions.", P))

    s.append(Paragraph("2. Validating the train/test data before splitting", H2))
    dq = pd.DataFrame([
        ["Mis-keyed posted_rate", "682 rows (1.42%), two clusters at ~0.29x and ~3.4x the expected rate",
         "Flagged with 5-fold out-of-fold residuals (|log residual| &gt; 0.35); removed from training only"],
        ["Negative weight", "292 train / 145 valid", "Sign error: residuals match |weight|, so use abs()"],
        ["Missing weight", "300 train / 165 valid", "Left as NaN (LightGBM native) + missing flag"],
        ["Weight = 47,500 lb", "1,204 train / 301 valid", "A cap, kept as a real value"],
        ["Missing market_index", "374 train / 249 valid", "Filled with that day's mean (index is ~daily: within-day sd 0.025 vs 0.167 across days)"],
        ["Unseen cities", "8 cities, 1,447 valid rows", "Cities enter via coordinates only, never as IDs"],
        ["Market shift", "mean market_index 1.08 train vs 0.93 valid", "Market effect modelled as an explicit elasticity, not left to trees"],
        ["Duplicates / bad coords / bad distance", "none found", "Coordinates constant per city; distance consistent per lane"],
    ], columns=["Check", "Finding", "Handling"])
    s.append(table(dq, widths=[1.35 * inch, 2.35 * inch, 3.3 * inch]))
    s.append(Spacer(1, 6))
    s.append(img(FIG / "label_outliers.png", 5.2 * inch))
    s.append(Paragraph("Out-of-fold residual ratios. The bulk is a tight peak; the two isolated clusters are mis-keyed rates. "
                       "Keeping them doubles backtest error (table below).", CAP))

    s.append(Paragraph("3. Split strategy", H2))
    s.append(Paragraph(
        "<b>Rolling-origin time split.</b> Each fold trains on everything before a cut-off and predicts the next two months, "
        "the same horizon as the real task. The real task observes October (the first month of Q4) and predicts November and "
        "December, so the two main folds have the same shape: <i>train through April, predict May-June</i>, and "
        "<i>train through July, predict August-September</i>. A third stress fold (train through August, predict "
        "September-October) also crosses a quarter boundary. All cleaning that uses labels (outlier filtering, smearing "
        "correction) is re-fitted inside each fold, so no test labels leak into training. Test-period mis-keyed labels are "
        "reported separately because no model can predict a typo.", P))
    s.append(Paragraph(
        "<b>Cold-city split.</b> 8 cities were removed from training entirely (every load touching them), and the model was "
        "scored on test loads that touch them. The cities drawn include a whole Texas/New Mexico cluster "
        "(Austin, Dallas, Amarillo, Lubbock, Albuquerque), which is harder than the real validation set, whose new cities are "
        "spread out.", P))

    s.append(Paragraph("4. Why the model is shaped this way", H2))
    s.append(Paragraph(
        "After controlling for lane, equipment and weight, rates follow the <b>7-day mean</b> of log market_index "
        "(elasticity about 0.15). The same-day value fits worse, because rates do not follow the index's strong weekly cycle. "
        "A clear <b>calendar pattern</b> also remains: rates climb through every quarter and reset on Apr 1, Jul 1 and Oct 1, "
        "and each quarter starts higher than the last. A tree model cannot extrapolate this into Nov-Dec. Worse, when trees "
        "were given day-level market features, they used them as a proxy for the calendar, and error on the later folds doubled. "
        "The final model therefore puts everything that is constant within a day into a small linear model: "
        "beta x 7-day market index + quarter level + convex within-quarter ramp. LightGBM models load-level structure: "
        "distance, coordinates, equipment, weight, and load-level deviations of market_index and quote_signal. "
        "The two parts are fitted by backfitting. The Q4 level is identified from October.", P))
    s.append(img(FIG / "time_structure.png", 6.6 * inch))

    s.append(Paragraph("5. Backtest results (clean test labels)", H2))
    bt_cols = ["Model"] + folds + ["Mean MAPE %", "Mean MAE $", "MAPE % (incl. bad labels)"]
    s.append(table(bt[bt_cols], widths=[1.55 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.75 * inch, 1.05 * inch], bold_row=0))
    s.append(Paragraph("MAPE = mean absolute % error on rate. The calendar terms matter most exactly where the validation set "
                       "sits (2nd-3rd month of a quarter). The 'incl. bad labels' column is dominated by the ~1.4% unpredictable "
                       "mis-keyed test labels.", CAP))
    cold_t = cold.assign(**{c: cold[c].map(lambda v: f"{v:.2f}") for c in ["MAE", "RMSE", "MAPE_%", "MedAPE_%"]})[["subset", "n", "MAE", "RMSE", "MAPE_%", "MedAPE_%"]]
    s.append(Spacer(1, 4))
    s.append(Paragraph("<b>Cold-city check</b> (stress fold, 8 cities hidden from training):", P))
    s.append(table(cold_t))
    s.append(Paragraph("Unseen cities cost roughly one extra point of MAPE in this deliberately harsh setup, and the model still beats the rate-card baseline on the same fold (3.19% vs 3.68% MAPE).", CAP))

    s.append(Paragraph("6. Final fit", H2))
    s.append(Paragraph("The final model is re-fitted on all 48,000 training loads, using the same cleaning and the same outlier rule. "
                       "It predicts all 12,000 validation loads, and every value is positive and finite.", P))

    s.append(KeepTogether([
        Paragraph("7. Fixed December prediction chart (produced by score.py)", H2),
        img(ROOT / "scorer_results" / "candidate_december.png", 6.8 * inch),
        Paragraph("Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb. The December input rows have no market_index or "
                  "quote_signal, so they get that day's market-wide values from validation.csv. The curve rises about 4% through "
                  "the month (about $846 to $882), which is the learned quarter-end ramp. It is smooth because rates follow the "
                  "7-day market mean, not the daily cycle. For reference, comparable Dry Van loads on this lane in training were "
                  "priced about $760-$935. Main assumption: the within-quarter ramp learned from Q1-Q3 repeats in Q4, whose level "
                  "is identified from October alone, so late-December rates carry the most uncertainty.", CAP),
    ]))

    SimpleDocTemplate(str(REP / "Freight_Rate_Report.pdf"), pagesize=letter,
                      leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch).build(s)
    print("wrote", REP / "Freight_Rate_Report.pdf")


if __name__ == "__main__":
    main()
