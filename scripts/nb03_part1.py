#!/usr/bin/env python3
"""
nb03_part1.py — Build notebooks/03_multivariate_var.ipynb
Sections 0–3: Einleitung, Datenladen, Stationaritätstest, Gemeinsamer Datensatz
"""
import nbformat

NB = "notebooks/03_multivariate_var.ipynb"


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


cells = []

# ════════════════════════════════════════════════════════════════════
# Section 0 — Einleitung und Motivation
# ════════════════════════════════════════════════════════════════════
cells.append(md("""\
# Multivariate Zeitreihenanalyse — VAR-Modell (Teil 3)

**Assets:** Gold (GC=F) · Bitcoin (BTC-USD) · EUR/USD (EURUSD=X)
**Zeitraum:** 2015-01-01 bis 2025-05-04 (eingefroren, reproduzierbar)\
"""))

cells.append(md("""\
---
## Einleitung und Motivation

### Ausgangslage: Ergebnis aus Teil 2

Die univariaten ARIMA-Analysen (Teil 2) haben für alle drei Assets dasselbe \
Grundmuster ergeben:

- **Gold:** ARIMA(0,1,1) schlägt den Random Walk nur marginal \
  (MASE > 1, OOS-MAE-Differenz < 0,3 %)
- **BTC & EUR/USD:** Ähnliches Bild — univariate Modelle liefern keinen \
  substanziellen Vorteil gegenüber dem Naiv-Forecast

Das bestätigt die **Effizienzmarkthypothese (EMH)**: Vergangene Renditen \
eines Assets enthalten kaum ausnutzbare Information über künftige Renditen \
desselben Assets.

### Neue Forschungsfrage

Wenn univariate Modelle scheitern — könnten **Wechselwirkungen zwischen \
den drei Märkten** zusätzliche Vorhersagekraft liefern?

Konkrete Hypothesen:
- Reagiert Gold auf Schocks in EUR/USD (Dollar-Effekt)?
- Bewegen sich BTC und Gold gemeinsam in Krisenzeiten (Safe-Haven-Muster)?
- Enthält der BTC-Markt Frühinformationen über Goldrenditen?

### Methode: VAR (Vector Autoregression)

Das **VAR(p)-Modell** schätzt alle drei Zeitreihen simultan:

$$\\mathbf{y}_t = \\mathbf{c} + \\boldsymbol{\\Phi}_1 \\mathbf{y}_{t-1} \
+ \\cdots + \\boldsymbol{\\Phi}_p \\mathbf{y}_{t-p} + \\boldsymbol{\\varepsilon}_t$$

mit $\\mathbf{y}_t = [r_{\\text{Gold},t},\\; r_{\\text{BTC},t},\\; \
r_{\\text{EUR/USD},t}]^\\top$ und $\\boldsymbol{\\varepsilon}_t \\sim \
\\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Sigma})$.

Jede Gleichung erklärt die Rendite eines Assets durch eigene Vergangenheitswerte \
**und** die Vergangenheitswerte der anderen Assets.\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 1 — Datenladen und visuelle Inspektion
# ════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Schritt 1: Datenladen und visuelle Inspektion\
"""))

cells.append(code("""\
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(42)

from src.utils.data_loader import load_series

def _fix_index(df):
    # Fix multi-level CSV header: convert index to DatetimeIndex, values to float.
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index.name = "Date"
    return df

# Load all three series (frozen 2015-01-01 to 2025-05-04)
gold   = _fix_index(load_series("GC=F"))
btc    = _fix_index(load_series("BTC-USD"))
eurusd = _fix_index(load_series("EURUSD=X"))

print(f"Gold   : {len(gold):,} Beobachtungen  "
      f"({gold.index[0].date()} bis {gold.index[-1].date()})")
print(f"BTC    : {len(btc):,} Beobachtungen  "
      f"({btc.index[0].date()} bis {btc.index[-1].date()})")
print(f"EUR/USD: {len(eurusd):,} Beobachtungen  "
      f"({eurusd.index[0].date()} bis {eurusd.index[-1].date()})")\
"""))

cells.append(code("""\
# Compute log-returns for all three assets
gold_ret   = np.log(gold["Close"]).diff().dropna()
btc_ret    = np.log(btc["Close"]).diff().dropna()
eurusd_ret = np.log(eurusd["Close"]).diff().dropna()

print(f"Log-Renditen berechnet — "
      f"Gold: {len(gold_ret)}, BTC: {len(btc_ret)}, EUR/USD: {len(eurusd_ret)}")\
"""))

# Plot 1: Price levels
cells.append(code("""\
# --- Plot 1: Price levels ---
C_GOLD = "#f59e0b"
C_BTC  = "#f97316"
C_EUR  = "#3b82f6"

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

for ax, series, label, color in zip(
    axes,
    [gold["Close"], btc["Close"], eurusd["Close"]],
    ["Gold (USD/Unze)", "Bitcoin (USD)", "EUR/USD"],
    [C_GOLD, C_BTC, C_EUR],
):
    ax.plot(series.index, series.values, color=color, linewidth=0.9)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Preisniveaus — Gold, Bitcoin, EUR/USD (2015–2025)",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()\
"""))

# Plot 2: Log-returns bar charts
cells.append(code("""\
# --- Plot 2: Log-returns (bar charts, green/red) ---
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

for ax, series, label, color_pos, color_neg in zip(
    axes,
    [gold_ret, btc_ret, eurusd_ret],
    ["Gold Log-Rendite", "BTC Log-Rendite", "EUR/USD Log-Rendite"],
    [C_GOLD, C_BTC, C_EUR],
    ["#dc2626", "#dc2626", "#dc2626"],
):
    colors = np.where(series.values >= 0, color_pos, color_neg)
    ax.bar(series.index, series.values, color=colors, width=1.0, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Tägliche Log-Renditen — Gold, Bitcoin, EUR/USD (2015–2025)",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()\
"""))

# Plot 3: Normalized prices
cells.append(code("""\
# --- Plot 3: Normalized prices (base = 100) ---
fig, ax = plt.subplots(figsize=(14, 5))

for series, label, color in zip(
    [gold["Close"], btc["Close"], eurusd["Close"]],
    ["Gold", "Bitcoin", "EUR/USD"],
    [C_GOLD, C_BTC, C_EUR],
):
    norm = series / series.iloc[0] * 100
    ax.plot(norm.index, norm.values, linewidth=1.2, label=label, color=color)

ax.axhline(100, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax.set_title("Normierte Preisentwicklung (Basis = 100, Jan 2015)", fontsize=14)
ax.set_ylabel("Index (Jan 2015 = 100)", fontsize=11)
ax.set_xlabel("Datum", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, ls="--", alpha=0.4)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
**Beobachtung (Datenladen & visuelle Inspektion):**

_(Hier kommt die Interpretation rein — Wie haben sich die drei Assets relativ \
entwickelt? Welches Asset zeigt die stärkste Preissteigerung? \
Sind Volatilitätsphasen sichtbar und gemeinsam?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 2 — Stationaritätstest
# ════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Schritt 2: Stationaritätstest aller drei Reihen

VAR setzt voraus, dass **alle Zeitreihen I(0)** sind (stationär). \
Wir testen ADF und KPSS für jede Log-Renditen-Reihe. \
Beide Tests müssen Stationarität bestätigen.\
"""))

cells.append(code("""\
from src.utils.data import adf_test, kpss_test

# Run ADF and KPSS on all three log-return series
stationarity_rows = []
for name, series in [("Gold", gold_ret), ("BTC", btc_ret), ("EUR/USD", eurusd_ret)]:
    adf  = adf_test(series,  name)
    kpss = kpss_test(series, name)
    stationarity_rows.append({
        "Asset"         : name,
        "ADF Stat."     : round(adf["ADF Teststatistik"], 3),
        "ADF p-Wert"    : adf["p-Wert"],
        "ADF Ergebnis"  : adf["Stationär (p < 0.05)"],
        "KPSS Stat."    : round(kpss["KPSS Teststatistik"], 3),
        "KPSS p-Wert"   : kpss["p-Wert"],
        "KPSS Ergebnis" : kpss["Stationär (p > 0.05)"],
    })

stat_df = pd.DataFrame(stationarity_rows)
stat_df\
"""))

cells.append(md("""\
**Interpretation der Stationaritätstests:**

VAR erfordert I(0)-Reihen. Da alle drei Log-Renditen-Reihen die \
Stationaritätsbedingung erfüllen (ADF stark signifikant, KPSS nicht \
signifikant), ist die VAR-Spezifikation auf Log-Renditen korrekt.

Ein mögliches KPSS-Grenzfallresultat (p ≈ 0,05) wäre durch \
**Heteroskedastizität** erklärbar (Volatilitäts-Cluster), nicht durch \
einen echten Unit-Root — analog zum Gold-Befund in Teil 2.

_(Hier kommt die konkrete Interpretation rein — stimmen ADF und KPSS überein?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 3 — Gemeinsamer Datensatz und deskriptive Statistik
# ════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Schritt 3: Gemeinsamer Datensatz und deskriptive Statistik

Wir synchronisieren alle drei Reihen auf einen gemeinsamen Datums-Index \
(inner join). Handelstage, an denen eines der Assets nicht gehandelt wird, \
fallen heraus.\
"""))

cells.append(code("""\
# Align all three return series on common dates (inner join)
df = pd.DataFrame({
    "Gold"   : gold_ret,
    "BTC"    : btc_ret,
    "EUR_USD": eurusd_ret,
}).dropna()
df.index = pd.to_datetime(df.index)

print(f"Gemeinsame Beobachtungen : {len(df):,}")
print(f"Zeitraum                 : {df.index[0].date()} bis {df.index[-1].date()}")
print(f"Fehlende Werte           : {df.isna().sum().to_dict()}")
print()
print(df.describe().round(6))\
"""))

# Correlation heatmap
cells.append(code("""\
# --- Correlation matrix heatmap ---
import matplotlib.colors as mcolors

corr = df.corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.3f}",
                ha="center", va="center", fontsize=12, fontweight="bold")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, fontsize=11)
ax.set_yticklabels(corr.index, fontsize=11)
ax.set_title("Korrelationsmatrix der Log-Renditen (2015–2025)", fontsize=13, pad=12)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()\
"""))

# Rolling 90-day correlation
cells.append(code("""\
# --- Rolling 90-day pairwise correlations ---
pairs = [("Gold", "BTC"), ("Gold", "EUR_USD"), ("BTC", "EUR_USD")]
pair_colors = [C_GOLD, C_EUR, C_BTC]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

for ax, (a, b), color in zip(axes, pairs, pair_colors):
    roll_corr = df[a].rolling(90).corr(df[b])
    ax.plot(roll_corr.index, roll_corr.values, color=color, linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.axhline(0.3, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(-0.3, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel(f"Korr. {a} / {b.replace('_', '/')}", fontsize=10)
    ax.set_ylim(-0.6, 0.8)
    ax.grid(True, ls="--", alpha=0.4)

axes[0].set_title("Rollende 90-Tage-Korrelationen der Log-Renditen",
                  fontsize=13, pad=10)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()\
"""))

# Descriptive stats table
cells.append(code("""\
# --- Descriptive statistics per asset ---
from scipy import stats as scipy_stats

desc_rows = []
for col in df.columns:
    s = df[col]
    jb_stat, jb_p = scipy_stats.jarque_bera(s.dropna())
    desc_rows.append({
        "Asset"         : col.replace("_", "/"),
        "Mittelwert"    : round(s.mean() * 100, 4),
        "Std. Abw. (%)" : round(s.std() * 100, 4),
        "Schiefe"       : round(float(scipy_stats.skew(s.dropna())), 4),
        "Exc. Kurtosis" : round(float(scipy_stats.kurtosis(s.dropna())), 4),
        "JB p-Wert"     : f"{jb_p:.2e}",
        "Min (%)"       : round(s.min() * 100, 4),
        "Max (%)"       : round(s.max() * 100, 4),
    })

pd.DataFrame(desc_rows).set_index("Asset")\
"""))

cells.append(md("""\
**Beobachtung (Deskriptive Analyse):**

_(Hier kommt die Interpretation rein — Wie korrelieren die drei Assets? \
Sind die Korrelationen stabil oder zeitlich variabel? \
Welches Asset hat die stärksten Heavy Tails?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Write Part 1
# ════════════════════════════════════════════════════════════════════
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

with open(NB, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Part 1 saved — {len(cells)} cells → {NB}")
