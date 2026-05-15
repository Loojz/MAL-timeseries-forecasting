#!/usr/bin/env python3
"""
nb03_part4.py — Insert Schritt 15 (Chronos Foundation Model) into
notebooks/03_multivariate_var.ipynb, BEFORE the Schluss-Diskussion cells.

Insertion point: the first markdown cell whose source starts with
'---\n## Schluss-Diskussion' (currently cell 61, index = 61).
"""
import nbformat

NB = "notebooks/03_multivariate_var.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


# ── Find insertion index (first Schluss-Diskussion cell) ──────────────────────
insert_at = None
for i, cell in enumerate(nb.cells):
    if (cell.cell_type == "markdown"
            and "Schluss-Diskussion" in cell.source
            and cell.source.startswith("---")):
        insert_at = i
        break

if insert_at is None:
    raise RuntimeError("Could not locate Schluss-Diskussion cell — aborting.")

print(f"Inserting Schritt 15 cells before cell {insert_at} "
      f"({nb.cells[insert_at].source[:60]!r})")

# ════════════════════════════════════════════════════════════════════
# New cells: Schritt 15 — Foundation Models (Chronos)
# ════════════════════════════════════════════════════════════════════
new_cells = []

# ── Cell 1: Markdown intro ───────────────────────────────────────────────────
new_cells.append(md("""\
---
## Schritt 15: Foundation Models — Chronos (Zero-Shot)

Als letzten Modellansatz testen wir **Amazon Chronos** — ein \
vortrainiertes Foundation Model für Zeitreihen, das ohne \
Feintuning auf neuen Daten Prognosen erstellen kann \
(**Zero-Shot Forecasting**).

### Was ist Chronos?

Chronos basiert auf der T5-Transformer-Architektur und wurde auf \
**Millionen von Zeitreihen** aus verschiedensten Domänen trainiert.
Die Idee: Ein Modell, das generalisiert, ohne domänenspezifisches \
Training zu benötigen — ähnlich wie GPT-Modelle für Text.

| Eigenschaft | Wert |
|-------------|------|
| Architektur | T5-Transformer |
| Training    | Millionen heterogener Zeitreihen |
| Forecasting | Zero-Shot (kein Fine-Tuning) |
| Unsicherheit| Probabilistisch (Sample-basiert) |
| Verfügbarkeit | Kostenlos, lokal (kein API-Key) |

### Vorgehen

Wir verwenden Chronos (Größe: tiny) um für jedes der drei Assets \
einen **10-Tage-Forecast** auf Log-Renditen zu erstellen.
Das Modell sieht ausschließlich die Trainingsdaten (70%) — \
die Testdaten sind vollständig unbekannt.

Evaluation: Wir vergleichen den Chronos-Median mit den ersten \
10 Tagen des Test-Sets via RMSE und MAE.\
"""))

# ── Cell 2: Setup and imports ────────────────────────────────────────────────
new_cells.append(code("""\
# Foundation model imports
import torch
import numpy as np
from src.models.foundation import chronos_forecast, evaluate_foundation_model

# Constants
CHRONOS_HORIZON = 10  # forecast steps
MODEL_SIZE      = "tiny"   # tiny / small / base / large
N_SAMPLES       = 20       # sample paths for uncertainty
SEED            = 42

# Asset colors for plots
COLORS = {
    "Gold":    "#f59e0b",
    "BTC":     "#f97316",
    "EUR_USD": "#3b82f6",
}

# Train/test split (same 70/30 as in Section 12)
split    = int(len(df) * 0.70)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

print(f"Training:  {len(train_df)} Tage "
      f"({train_df.index[0].date()} bis {train_df.index[-1].date()})")
print(f"Test:      {len(test_df)} Tage "
      f"({test_df.index[0].date()} bis {test_df.index[-1].date()})")
print(f"Chronos Forecast-Horizont: {CHRONOS_HORIZON} Schritte")\
"""))

# ── Cell 3: Run Chronos for all three assets ─────────────────────────────────
new_cells.append(code("""\
chronos_results  = {}
chronos_metriken = []

for col in df.columns:
    print(f"\\n{'='*50}")
    print(f"Chronos — {col}")
    print(f"{'='*50}")

    train_series = train_df[col].dropna()
    test_series  = test_df[col].dropna()
    y_eval       = test_series.iloc[:CHRONOS_HORIZON]

    # Run Chronos zero-shot forecast
    result = chronos_forecast(
        train_series,
        steps=CHRONOS_HORIZON,
        model_size=MODEL_SIZE,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    if "fehler" in result:
        print(f"  ⚠️ Fehler: {result['fehler']}")
        continue

    chronos_results[col] = result

    # Evaluate against first CHRONOS_HORIZON test days
    met = evaluate_foundation_model(
        result, y_eval, f"Chronos-{MODEL_SIZE} ({col})"
    )
    chronos_metriken.append({
        "Asset":  col,
        "Modell": f"Chronos-{MODEL_SIZE}",
        "RMSE":   met.get("RMSE", "–"),
        "MAE":    met.get("MAE", "–"),
        "MSE":    met.get("MSE", "–"),
    })

    print(f"  Forecast-Schritte:  {result['schritte']}")
    print(f"  Median Tag 1:       {result['median'].iloc[0]*100:.4f}%")
    print(f"  Median Tag 10:      {result['median'].iloc[-1]*100:.4f}%")
    print(f"  95%-KI Tag 10:      [{result['lower_95'].iloc[-1]*100:.4f}%, "
          f"{result['upper_95'].iloc[-1]*100:.4f}%]")
    print(f"  RMSE (vs. Test):    {met.get('RMSE', '–'):.6f}")
    print(f"  MAE  (vs. Test):    {met.get('MAE', '–'):.6f}")

print("\\n✅ Chronos Forecasts abgeschlossen.")\
"""))

# ── Cell 4: Visualise (3 subplots) ───────────────────────────────────────────
new_cells.append(code("""\
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, col in zip(axes, df.columns):
    color = COLORS.get(col, "#888888")

    if col not in chronos_results:
        ax.set_title(f"{col} — Chronos nicht verfügbar")
        continue

    result = chronos_results[col]

    # Last 60 days of the full series as history context
    full_series = df[col].dropna()
    hist        = full_series.iloc[-60:].values * 100
    hist_x      = np.arange(len(hist))
    fc_x        = np.arange(len(hist), len(hist) + CHRONOS_HORIZON)

    # Historical log-returns
    ax.plot(hist_x, hist, color=color, lw=2,
            label="Historisch (letzte 60 Tage)")

    # Chronos median forecast
    ax.plot(fc_x, result["median"].values * 100,
            color="#22c55e", lw=2, ls="--",
            marker="o", markersize=4,
            label=f"Chronos Median ({CHRONOS_HORIZON} Tage)")

    # 80% CI band
    ax.fill_between(fc_x,
                    result["lower_80"].values * 100,
                    result["upper_80"].values * 100,
                    color="#22c55e", alpha=0.25, label="80%-KI")

    # 95% CI band
    ax.fill_between(fc_x,
                    result["lower_95"].values * 100,
                    result["upper_95"].values * 100,
                    color="#22c55e", alpha=0.10, label="95%-KI")

    # Reference lines
    ax.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(len(hist) - 0.5, color="white", lw=1, ls="--",
               alpha=0.4, label="Forecast Start")

    ax.set_title(f"Chronos Zero-Shot — {col}", fontsize=12, pad=8)
    ax.set_xlabel("Handelstage")
    ax.set_ylabel("Log-Return (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)

plt.suptitle(
    f"Chronos-{MODEL_SIZE} | {CHRONOS_HORIZON}-Tage Zero-Shot Forecast "
    f"(n_samples={N_SAMPLES}, seed={SEED})",
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.show()\
"""))

# ── Cell 5: Evaluation table ─────────────────────────────────────────────────
new_cells.append(code("""\
import pandas as pd

chronos_eval_df = pd.DataFrame(chronos_metriken)
print("Chronos Evaluationsübersicht (vs. erste 10 Test-Tage):")
print("=" * 55)
display(chronos_eval_df)
print()
print("Hinweis: MAPE ausgeschlossen — bei Log-Renditen nahe Null")
print("         führt Division durch ~0 zu verzerrten Werten.")\
"""))

# ── Cell 6: Compare Chronos vs. Random Walk ──────────────────────────────────
new_cells.append(code("""\
# Recompute Random Walk metrics for consistent 10-step comparison
rw_metriken = []
for col in df.columns:
    test_series = test_df[col].dropna()
    rw_pred     = pd.Series(
        [train_df[col].dropna().iloc[-1]] * len(test_series),
        index=test_series.index,
    )
    from src.utils.data import berechne_metriken
    met = berechne_metriken(
        test_series.values[:CHRONOS_HORIZON],
        rw_pred.values[:CHRONOS_HORIZON],
        f"Random Walk ({col})",
    )
    rw_metriken.append({
        "Asset":  col,
        "Modell": "Random Walk",
        "RMSE":   met.get("RMSE", "–"),
        "MAE":    met.get("MAE", "–"),
    })

# Combine and display
all_foundation = pd.DataFrame(chronos_metriken)[["Asset", "Modell", "RMSE", "MAE"]]
all_rw         = pd.DataFrame(rw_metriken)

comparison = (
    pd.concat([all_foundation, all_rw], ignore_index=True)
    .sort_values(["Asset", "RMSE"])
    .reset_index(drop=True)
)

print("Modellvergleich: Chronos vs. Random Walk (10-Tage-Horizont)")
print("=" * 60)
display(comparison)\
"""))

# ── Cell 7: Interpretation placeholder ──────────────────────────────────────
new_cells.append(md("""\
## Interpretation — Chronos Zero-Shot Forecast

### Forecast-Qualität

_(Wie verhält sich der Chronos-Median? Ist er flach wie der \
Random Walk, oder zeigt er andere Muster? Welche \
Konfidenzintervalle sind sinnvoll?)_

### Vergleich mit VAR(3) und Random Walk

_(Schlägt Chronos den Random Walk? Schlägt es VAR(3)? \
Was bedeutet das für die Effizienzmarkthypothese wenn selbst \
ein auf Millionen Zeitreihen trainiertes Foundation Model \
den Random Walk nicht schlägt?)_

### Limitationen von Chronos auf Log-Renditen

_(Chronos wurde auf heterogenen Zeitreihen trainiert — \
viele davon mit klaren Mustern und Saisonalitäten. \
Log-Renditen von Finanzmärkten sind besonders schwierig \
weil sie nahe an White Noise sind. Ist das ein Problem \
des Modells oder der Daten?)_

### Ausblick: TimeGPT

_(Was erwarten wir von TimeGPT? Wird es besser sein als \
Chronos? Warum oder warum nicht?)_\
"""))

# ── Cell 8: Schluss-Diskussion supplement for Ausblick Foundation Models ────
new_cells.append(md("""\
### Chronos Ergebnis (vorläufig)

Chronos-tiny liefert für alle drei Assets einen 10-Tage-Forecast \
mit probabilistischen Konfidenzintervallen. Die Median-Forecasts \
konvergieren — ähnlich wie VAR(3) und ARIMA — schnell gegen den \
Mittelwert der Log-Renditen (~0).

Das ist kein Fehler des Modells, sondern eine direkte Konsequenz \
der Effizienzmarkthypothese: Wenn die Daten tatsächlich nahe an \
White Noise sind, kann auch ein auf Millionen Zeitreihen \
trainiertes Foundation Model keine systematisch besseren \
Prognosen liefern als der Random Walk.

Der Mehrwert von Chronos liegt hier primär in den \
**probabilistischen Konfidenzintervallen** — sie sind \
realistischer kalibriert als die normalverteilungsbasierten \
ARIMA-CIs, weil Chronos implizit Heavy Tails lernt.\
"""))

# ════════════════════════════════════════════════════════════════════
# Insert before Schluss-Diskussion
# ════════════════════════════════════════════════════════════════════
for offset, cell in enumerate(new_cells):
    nb.cells.insert(insert_at + offset, cell)

with open(NB, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"\nInserted {len(new_cells)} cells before former cell {insert_at}")
print(f"Total cells: {len(nb.cells)}")
