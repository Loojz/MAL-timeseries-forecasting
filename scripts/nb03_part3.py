#!/usr/bin/env python3
"""
nb03_part3.py — Append Sections 12–14 to notebooks/03_multivariate_var.ipynb
Backtesting, 10-Schritt-Prognose, Schluss-Diskussion
"""
import nbformat

NB = "notebooks/03_multivariate_var.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


new_cells = []

# ════════════════════════════════════════════════════════════════════
# Section 12 — Backtesting
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 12: Backtesting — VAR vs. ARIMA vs. Random Walk

Die zentrale Frage: **Schlägt VAR den Random-Walk-Benchmark aus Teil 2?**

Wir verwenden einen 70/30-Train-Test-Split und vergleichen vier Modelle \
auf allen drei Assets:\
"""))

new_cells.append(code("""\
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from src.utils.data import berechne_metriken

# 70/30 split
split = int(len(df) * 0.70)
train = df.iloc[:split]
test  = df.iloc[split:]

print(f"Train: {len(train):,} Obs  ({train.index[0].date()} bis {train.index[-1].date()})")
print(f"Test : {len(test):,} Obs  ({test.index[0].date()} bis {test.index[-1].date()})")\
"""))

new_cells.append(code("""\
# --- Fit all four models on train ---

# VAR(p_auto)
var_auto_tr = VAR(train).fit(p_auto)
var_auto_fc = var_auto_tr.forecast(
    train.values[-max(p_auto, 1):], steps=len(test)
)
var_auto_fc_df = pd.DataFrame(var_auto_fc, index=test.index, columns=df.columns)

# VAR(3)
var3_tr = VAR(train).fit(3)
var3_fc = var3_tr.forecast(train.values[-3:], steps=len(test))
var3_fc_df = pd.DataFrame(var3_fc, index=test.index, columns=df.columns)

# MA(1) per asset on log-returns (= ARIMA(0,1,1) on price levels, returns-equivalent)
ma1_fc = {}
for col in df.columns:
    mod = ARIMA(train[col], order=(0, 0, 1)).fit()
    ma1_fc[col] = mod.forecast(steps=len(test)).values

# Random Walk — predict the unconditional mean of training returns
rw_fc = {col: np.full(len(test), train[col].mean()) for col in df.columns}

print("Alle vier Modelle gefittet.")\
"""))

new_cells.append(code("""\
# --- Compute metrics for all models ---
all_metrics = []

for col in df.columns:
    y_true = test[col].values
    for model_name, y_pred in [
        (f"VAR({p_auto})", var_auto_fc_df[col].values),
        ("VAR(3)",         var3_fc_df[col].values),
        ("MA(1)",          ma1_fc[col]),
        ("Random Walk",    rw_fc[col]),
    ]:
        m = berechne_metriken(y_true, y_pred, model_name)
        m["Asset"] = col.replace("_", "/")
        all_metrics.append(m)

master = pd.DataFrame(all_metrics)[[
    "Asset", "Modell", "MSE", "RMSE", "MAE", "MAPE (%)"
]].sort_values(["Asset", "RMSE"]).reset_index(drop=True)

# Display per asset
for asset in master["Asset"].unique():
    print(f"\\n=== {asset} ===")
    print(master[master["Asset"] == asset].drop("Asset", axis=1).to_string(index=False))\
"""))

new_cells.append(code("""\
# --- Forecast plots: actual vs. predicted per asset ---
C_GOLD, C_BTC, C_EUR = "#f59e0b", "#f97316", "#3b82f6"

for col, asset_label, base_color in zip(
    df.columns,
    ["Gold", "BTC", "EUR/USD"],
    [C_GOLD, C_BTC, C_EUR],
):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Last 60 days of history + full test
    hist_start = train.index[-60]
    ax.plot(df.loc[hist_start:train.index[-1], col].index,
            df.loc[hist_start:train.index[-1], col].values,
            color=base_color, linewidth=1.2, label="Aktuelle Werte (Train)")
    ax.plot(test[col].index, test[col].values,
            color=base_color, linewidth=1.5, label="Aktuelle Werte (Test)",
            linestyle="-")

    styles = [("--", "#1d4ed8"), (":", "#7c3aed"), ("-.", "#dc2626"), ("--", "#6b7280")]
    for (fc_name, fc_vals), (ls, fc_color) in zip(
        [
            (f"VAR({p_auto})", var_auto_fc_df[col].values),
            ("VAR(3)",         var3_fc_df[col].values),
            ("MA(1)",          ma1_fc[col]),
            ("Random Walk",    rw_fc[col]),
        ],
        styles,
    ):
        ax.plot(test.index, fc_vals, color=fc_color, linewidth=1.0,
                linestyle=ls, alpha=0.8, label=fc_name)

    ax.axvline(test.index[0], color="black", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.set_title(f"Backtesting — {asset_label} Log-Renditen",
                 fontsize=13, pad=10)
    ax.set_ylabel("Log-Rendite", fontsize=10)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()\
"""))

new_cells.append(code("""\
# --- Winner table per asset ---
best_rows = []
for col, asset_label in zip(df.columns, ["Gold", "BTC", "EUR/USD"]):
    sub = master[master["Asset"] == col.replace("_", "/")].copy()
    best = sub.loc[sub["RMSE"].idxmin()]
    rw_rmse = float(sub[sub["Modell"] == "Random Walk"]["RMSE"].values[0])
    improvement = (rw_rmse - float(best["RMSE"])) / rw_rmse * 100
    best_rows.append({
        "Asset"                    : asset_label,
        "Bestes Modell (RMSE)"     : best["Modell"],
        "RMSE (Best)"              : round(float(best["RMSE"]), 8),
        "RMSE (Random Walk)"       : round(rw_rmse, 8),
        "Verbesserung ggü. RW (%)" : f"{improvement:+.2f}%",
    })

winner_df = pd.DataFrame(best_rows)
print("=== Gewinnermodell je Asset ===")
print(winner_df.to_string(index=False))\
"""))

new_cells.append(md("""\
**Beobachtung (Backtesting):**

_(Hier kommt die Interpretation rein — schlägt VAR den Random Walk? \
Bei welchem Asset ist der Unterschied am größten? \
Ist die Verbesserung praktisch relevant oder statistisch marginal?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 13 — 10-Schritt-Prognose
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 13: 10-Schritt-Prognose mit VAR(3)

Wir fitten VAR(3) auf dem **vollständigen Datensatz** und erstellen eine \
Prognose für die nächsten 10 Handelstage.\
"""))

new_cells.append(code("""\
from src.models.var_model import var_prognose

# Fit VAR(3) on full data
var3_full = VAR(df).fit(3)

# 10-step forecast
forecast_result = var_prognose(var3_full, steps=10)

fc_df    = forecast_result["prognose"]
lower_df = forecast_result["lower_95"]
upper_df = forecast_result["upper_95"]

print("=== 10-Schritt VAR(3)-Prognose (Log-Renditen) ===")
print(fc_df.round(6))\
"""))

new_cells.append(code("""\
# --- Plot forecast for each asset ---
import matplotlib.dates as mdates
import pandas as pd

# Build future date index (10 business days after last date)
last_date  = df.index[-1]
future_idx = pd.bdate_range(start=last_date, periods=11, freq="B")[1:]

for i, (col, asset_label, base_color) in enumerate(zip(
    df.columns,
    ["Gold", "BTC", "EUR/USD"],
    [C_GOLD, C_BTC, C_EUR],
)):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Last 90 days of historical returns
    hist = df[col].iloc[-90:]
    ax.plot(hist.index, hist.values, color=base_color,
            linewidth=1.0, label="Historische Log-Renditen")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # Forecast
    fc_vals = fc_df.iloc[:, i].values
    lo_vals = lower_df.iloc[:, i].values
    hi_vals = upper_df.iloc[:, i].values

    ax.plot(future_idx, fc_vals, color="#1d4ed8", linewidth=2.0,
            marker="o", markersize=5, label="VAR(3) Prognose")
    ax.fill_between(future_idx, lo_vals, hi_vals,
                    alpha=0.25, color="#1d4ed8", label="95%-KI")

    ax.axvline(last_date, color="black", linewidth=0.8,
               linestyle="--", alpha=0.6)
    ax.set_title(f"10-Schritt-Prognose — {asset_label} (Log-Renditen)",
                 fontsize=13, pad=10)
    ax.set_ylabel("Log-Rendite", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()\
"""))

new_cells.append(code("""\
# Forecast table
fc_table = fc_df.copy()
fc_table.index = future_idx
fc_table.columns = [c.replace("_", "/") for c in fc_table.columns]
fc_table.index.name = "Datum"
print("=== VAR(3) Prognose-Tabelle (Log-Renditen) ===")
print(fc_table.round(6))\
"""))

new_cells.append(md("""\
**Beobachtung (10-Schritt-Prognose):**

_(Hier kommt die Interpretation rein — wie weit divergieren die Prognosen \
der drei Assets? Nähern sich die Prognosen über die Zeit dem \
Mittelwert an? Wie breit sind die Konfidenzintervalle?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 14 — Schluss-Diskussion
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schluss-Diskussion: Teil 3\
"""))

new_cells.append(md("""\
### Hauptergebnis

_(Schlägt VAR(3) den Random Walk? Gibt es Granger-Kausalitäten? \
Was zeigen die IRFs und die FEVD? Lautet das Gesamturteil: \
"Multivariate Modelle bringen Mehrwert" oder "Random Walk bleibt schwer schlagbar"?)_

---

### Vergleich mit Teil 2 (univariate ARIMA)

_(Bringt der multivariate Ansatz Mehrwert gegenüber den univariaten Modellen \
aus Teil 2? Liegt die Verbesserung im RMSE über dem Konfidenzband der \
Messungenauigkeit?)_

---

### Granger-Kausalitäten — ökonomische Interpretation

_(Welche Märkte beeinflussen sich? Was bedeutet eine signifikante \
Gold → BTC Granger-Kausalität oder EUR/USD → Gold inhaltlich? \
Gibt es einen Dollar-Transmissionskanal?)_

---

### Limitationen

_(Was kann VAR nicht?
— Nicht-lineare Abhängigkeiten (Regime-Switching)
— Keine Volatilitätsmodellierung (GARCH-DCC wäre der nächste Schritt)
— Cholesky-Identifikation ist sensitiv gegenüber der Reihenfolge
— Heavy Tails in allen drei Residuen — Normalverteilungs-CIs zu eng)_

---

### Ausblick: Foundation Models

_(Chronos (Amazon) und TimeGPT (Nixtla) als nächster Schritt: \
pre-trainierte Transformer-Modelle, die zero-shot Prognosen liefern. \
Hypothese: Schlagen sie den VAR-Benchmark aus Teil 3 auf dem \
30-Tage-Horizont? Oder bleibt der Random Walk unschlagbar?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Write Part 3
# ════════════════════════════════════════════════════════════════════
nb.cells.extend(new_cells)

with open(NB, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Part 3 saved — added {len(new_cells)} cells")
print(f"Total cells: {len(nb.cells)}")
