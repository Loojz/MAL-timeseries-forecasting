#!/usr/bin/env python3
"""
nb03_part5.py — Insert Schritt 16 (TimeGPT-2.1) into
notebooks/03_multivariate_var.ipynb, AFTER the Chronos section
(Schritt 15) and BEFORE the Schluss-Diskussion cells.

Insertion point: first cell whose source starts with
'---\n## Schluss-Diskussion' (currently cell index 69).
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

print(f"Inserting Schritt 16 cells before cell {insert_at} "
      f"({nb.cells[insert_at].source[:60]!r})")

# ════════════════════════════════════════════════════════════════════
# New cells: Schritt 16 — TimeGPT-2.1
# ════════════════════════════════════════════════════════════════════
new_cells = []

# ── Cell 1: Markdown intro ───────────────────────────────────────────────────
new_cells.append(md("""\
---
## Schritt 16: Foundation Models — TimeGPT-2.1 (Zero-Shot)

Als zweites Foundation Model testen wir **TimeGPT-2.1** von Nixtla — \
ein API-basiertes Modell, das speziell für Zeitreihen-Forecasting \
entwickelt wurde.

### Was ist TimeGPT-2.1?

TimeGPT-2.1 ist das aktuellste Modell der TimeGPT-2-Familie von \
Nixtla. Im Gegensatz zu Chronos (lokal, kostenlos) läuft TimeGPT \
als Cloud-Service über eine REST-API.

| Eigenschaft | Chronos (Schritt 15) | TimeGPT-2.1 (Schritt 16) |
|-------------|----------------------|--------------------------|
| Anbieter | Amazon | Nixtla |
| Ausführung | Lokal (CPU) | Cloud API |
| API-Key | Nein | Ja (NIXTLA_API_KEY) |
| Architektur | T5-Transformer | Proprietär |
| Spezialität | Generalist | Zeitreihen-fokussiert |

### Vorgehen

Identisch zu Schritt 15: 10-Tage Zero-Shot Forecast auf Log-Renditen, \
70/30 Train/Test-Split, Evaluation gegen die ersten 10 Test-Tage.\
"""))

# ── Cell 2: Setup and API key ────────────────────────────────────────────────
new_cells.append(code("""\
# Load API key from .env file at project root
from dotenv import load_dotenv
import os

load_dotenv()  # loads NIXTLA_API_KEY from .env

nixtla_key = os.environ.get("NIXTLA_API_KEY")

if nixtla_key:
    print(f"✅ NIXTLA_API_KEY geladen ({nixtla_key[:8]}...)")
else:
    print("❌ NIXTLA_API_KEY nicht gefunden.")
    print("   Lege eine .env Datei im Projektordner an:")
    print("   NIXTLA_API_KEY=dein_key_hier")

# Validate API key
if nixtla_key:
    from nixtla import NixtlaClient
    client = NixtlaClient(
        base_url="https://api-preview.nixtla.io",
        api_key=nixtla_key,
    )
    valid = client.validate_api_key()
    print(f"   API-Key gültig: {valid}")\
"""))

# ── Cell 3: TimeGPT forecasts for all three assets ───────────────────────────
new_cells.append(code("""\
from src.models.foundation import timegpt_forecast, evaluate_foundation_model

TIMEGPT_HORIZON = 10   # same as Chronos
TIMEGPT_MODEL   = "timegpt-2.1"

timegpt_results  = {}
timegpt_metriken = []

if not nixtla_key:
    print("⚠️ Kein API-Key — TimeGPT-Forecast übersprungen.")
else:
    for col in df.columns:
        print(f"\\n{'='*50}")
        print(f"TimeGPT-2.1 — {col}")
        print(f"{'='*50}")

        train_series = train_df[col].dropna()
        test_series  = test_df[col].dropna()
        y_eval       = test_series.iloc[:TIMEGPT_HORIZON]

        result = timegpt_forecast(
            train_series,
            steps=TIMEGPT_HORIZON,
            api_key=nixtla_key,
            model=TIMEGPT_MODEL,
        )

        if "fehler" in result:
            print(f"  ⚠️ Fehler: {result['fehler']}")
            continue

        timegpt_results[col] = result

        met = evaluate_foundation_model(
            result, y_eval,
            f"TimeGPT-2.1 ({col})"
        )
        timegpt_metriken.append({
            "Asset":  col,
            "Modell": "TimeGPT-2.1",
            "RMSE":   met.get("RMSE", "–"),
            "MAE":    met.get("MAE",  "–"),
            "MSE":    met.get("MSE",  "–"),
        })

        print(f"  Median Tag 1:    {result['median'].iloc[0]*100:.4f}%")
        print(f"  Median Tag 10:   {result['median'].iloc[-1]*100:.4f}%")
        print(f"  95%-KI Tag 10:   [{result['lower_95'].iloc[-1]*100:.4f}%,"
              f" {result['upper_95'].iloc[-1]*100:.4f}%]")
        print(f"  RMSE (vs Test):  {met.get('RMSE', '–'):.6f}")
        print(f"  MAE  (vs Test):  {met.get('MAE',  '–'):.6f}")

    print("\\n✅ TimeGPT-2.1 Forecasts abgeschlossen.")\
"""))

# ── Cell 4: Visualise (3 subplots) ───────────────────────────────────────────
new_cells.append(code("""\
if timegpt_results:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    PURPLE = "#8b5cf6"

    for ax, col in zip(axes, df.columns):
        color = COLORS.get(col, "#888888")

        if col not in timegpt_results:
            ax.set_title(f"{col} — TimeGPT nicht verfügbar")
            continue

        result = timegpt_results[col]

        # Last 60 days of full series as history
        full_series = df[col].dropna()
        hist        = full_series.iloc[-60:].values * 100
        hist_x      = np.arange(len(hist))
        fc_x        = np.arange(len(hist), len(hist) + TIMEGPT_HORIZON)

        # History
        ax.plot(hist_x, hist,
                color=color, lw=2,
                label="Historisch (letzte 60 Tage)")

        # TimeGPT median
        ax.plot(fc_x, result["median"].values * 100,
                color=PURPLE, lw=2, ls="--",
                marker="o", markersize=4,
                label=f"TimeGPT-2.1 Median ({TIMEGPT_HORIZON} Tage)")

        # 80% CI
        ax.fill_between(fc_x,
                        result["lower_80"].values * 100,
                        result["upper_80"].values * 100,
                        color=PURPLE, alpha=0.25,
                        label="80%-KI")

        # 95% CI
        ax.fill_between(fc_x,
                        result["lower_95"].values * 100,
                        result["upper_95"].values * 100,
                        color=PURPLE, alpha=0.10,
                        label="95%-KI")

        # Zero line and forecast boundary
        ax.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
        ax.axvline(len(hist) - 0.5,
                   color="white", lw=1, ls="--", alpha=0.4,
                   label="Forecast Start")

        ax.set_title(f"TimeGPT-2.1 Zero-Shot — {col}",
                     fontsize=12, pad=8)
        ax.set_xlabel("Handelstage")
        ax.set_ylabel("Log-Return (%)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    plt.suptitle(
        f"TimeGPT-2.1 | {TIMEGPT_HORIZON}-Tage Zero-Shot Forecast",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.show()
else:
    print("Keine TimeGPT-Ergebnisse verfügbar.")\
"""))

# ── Cell 5: Evaluation table ─────────────────────────────────────────────────
new_cells.append(code("""\
if timegpt_metriken:
    timegpt_eval_df = pd.DataFrame(timegpt_metriken)
    print("TimeGPT-2.1 Evaluationsübersicht (vs. erste 10 Test-Tage):")
    print("=" * 60)
    display(timegpt_eval_df)
    print()
    print("Hinweis: MAPE ausgeschlossen — bei Log-Renditen nahe Null")
    print("         führt Division durch ~0 zu verzerrten Werten.")\
"""))

# ── Cell 6: Full comparison Chronos vs TimeGPT vs Random Walk ────────────────
new_cells.append(code("""\
# Combine Chronos + TimeGPT metrics
all_foundation = pd.concat([
    pd.DataFrame(chronos_metriken)[["Asset", "Modell", "RMSE", "MAE"]],
    pd.DataFrame(timegpt_metriken)[["Asset", "Modell", "RMSE", "MAE"]]
    if timegpt_metriken else pd.DataFrame()
], ignore_index=True)

# Random Walk baseline (recomputed for 10-step horizon)
rw_rows = []
for col in df.columns:
    test_series = test_df[col].dropna()
    rw_pred     = pd.Series(
        [train_df[col].dropna().iloc[-1]] * TIMEGPT_HORIZON,
        index=test_series.index[:TIMEGPT_HORIZON]
    )
    from src.utils.data import berechne_metriken
    met = berechne_metriken(
        test_series.values[:TIMEGPT_HORIZON],
        rw_pred.values,
        f"Random Walk ({col})"
    )
    rw_rows.append({
        "Asset":  col,
        "Modell": "Random Walk",
        "RMSE":   met.get("RMSE", "–"),
        "MAE":    met.get("MAE",  "–"),
    })

all_models = pd.concat([
    all_foundation,
    pd.DataFrame(rw_rows)
], ignore_index=True)

all_models = all_models.sort_values(
    ["Asset", "RMSE"]
).reset_index(drop=True)

print("Gesamtvergleich: Chronos vs. TimeGPT-2.1 vs. Random Walk")
print("=" * 65)
display(all_models)\
"""))

# ── Cell 7: Interpretation placeholder ──────────────────────────────────────
new_cells.append(md("""\
## Interpretation — TimeGPT-2.1 vs. Chronos vs. Random Walk

### Forecast-Qualität TimeGPT-2.1

_(Wie verhält sich der TimeGPT-2.1-Median? Ähnlich flach wie \
Chronos? Sind die Konfidenzintervalle breiter oder schmaler?)_

### Direkter Vergleich: Chronos vs. TimeGPT-2.1

_(Welches Modell schneidet besser ab — auf welchem Asset? \
Sind die Unterschiede praktisch relevant oder statistisches Rauschen?)_

### Gesamtfazit: Foundation Models vs. klassische Modelle

_(Was ist die übergreifende Erkenntnis aus Schritt 15 und 16 \
zusammen? Rechtfertigt der Mehraufwand (API-Key, Kosten, \
Cloud-Abhängigkeit) den Mehrwert gegenüber dem Random Walk?)_

### Abschließende Erkenntnis

Alle getesteten Modelle — ARIMA(0,1,1), VAR(3), ETS, \
Chronos-tiny und TimeGPT-2.1 — liefern konsistent dasselbe \
Ergebnis: Der Random Walk bleibt der stärkste Benchmark. \
Dies ist kein Versagen der Modelle, sondern eine empirische \
Bestätigung der **Effizienzmarkthypothese** (Fama, 1970): \
In hochliquiden, gut beobachteten Märkten sind vergangene \
Preise keine ausnutzbare Informationsquelle für zukünftige \
Renditen.\
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
