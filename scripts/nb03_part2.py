#!/usr/bin/env python3
"""
nb03_part2.py — Append Sections 4–11 to notebooks/03_multivariate_var.ipynb
Kointegration, QLR, Lag-Selektion, VAR-Fit, Granger, IRF, FEVD
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
# Section 4 — Kointegrations-Test (Johansen)
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 4: Kointegrations-Test (Johansen)

Bevor wir VAR fitten, müssen wir klären: **VAR auf Renditen oder VECM auf \
Niveaus?**

### Konzept: Kointegration

Zwei I(1)-Reihen sind **kointegriert**, wenn eine Linearkombination davon \
stationär (I(0)) ist — d. h. sie teilen einen gemeinsamen Langzeit-Trend.

| Ergebnis | Konsequenz |
|----------|------------|
| Kointegration vorhanden | → **VECM** (berücksichtigt Gleichgewichtskorrektur) |
| Keine Kointegration | → **VAR auf Log-Renditen** (I(0)) ist korrekt |

Der **Johansen-Test** prüft die Anzahl kointegrierender Vektoren $r$ \
(Rang der Kointegrationsmatrix).\
"""))

new_cells.append(code("""\
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Log-price levels (non-stationary I(1) series) — common dates only
log_prices = pd.DataFrame({
    "Gold"   : np.log(gold["Close"]),
    "BTC"    : np.log(btc["Close"]),
    "EUR_USD": np.log(eurusd["Close"]),
}).dropna()
log_prices.index = pd.to_datetime(log_prices.index)
log_prices = log_prices[log_prices.index.isin(df.index)]

johansen = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
n_vars = log_prices.shape[1]

# Trace-Statistik table
trace_table = pd.DataFrame({
    "H0"               : [f"r ≤ {i}" for i in range(n_vars)],
    "Trace-Statistik"  : johansen.lr1.round(3),
    "Krit. Wert 90%"   : johansen.cvt[:, 0].round(3),
    "Krit. Wert 95%"   : johansen.cvt[:, 1].round(3),
    "Signifikant (95%)": ["✓" if johansen.lr1[i] > johansen.cvt[i, 1]
                          else "✗" for i in range(n_vars)],
})

# Max-Eigenvalue table
max_table = pd.DataFrame({
    "H0"               : [f"r = {i}" for i in range(n_vars)],
    "Max-Eigen-Stat."  : johansen.lr2.round(3),
    "Krit. Wert 90%"   : johansen.cvm[:, 0].round(3),
    "Krit. Wert 95%"   : johansen.cvm[:, 1].round(3),
    "Signifikant (95%)": ["✓" if johansen.lr2[i] > johansen.cvm[i, 1]
                          else "✗" for i in range(n_vars)],
})

print("=== Johansen Trace-Test ===")
print(trace_table.to_string(index=False))
print()
print("=== Johansen Max-Eigenvalue-Test ===")
print(max_table.to_string(index=False))\
"""))

new_cells.append(md("""\
**Schlussfolgerung — VAR vs. VECM:**

Wenn kein kointegrierender Vektor signifikant ist (alle Trace-Statistiken \
< Krit. Wert 95 %), ist **kein gemeinsamer Langzeit-Trend** nachweisbar. \
→ **VAR auf Log-Renditen ist methodisch korrekt.**

Falls ein Rang r > 0 nachgewiesen wird, wäre VECM die theoretisch sauberere \
Wahl — für diese Arbeit bleibt VAR auf I(0)-Renditen die primäre Methode.

_(Hier kommt die konkrete Interpretation rein — wie viele kointegrierte \
Vektoren wurden gefunden?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 5 — QLR-Test
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 5: QLR-Test — Strukturbrüche ohne bekannten Bruchpunkt

Der **QLR-Test (Andrews 1993)** sucht den Bruchpunkt $T^*$ der den \
F-Test-Wert maximiert — ohne vorab einen Zeitpunkt angeben zu müssen. \
Wir testen jede der drei Log-Renditen-Reihen separat.\
"""))

new_cells.append(code("""\
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.models.arima_model import qlr_test

qlr_results = {}
for name, series in [("Gold", gold_ret), ("BTC", btc_ret), ("EUR/USD", eurusd_ret)]:
    qlr_results[name] = qlr_test(series)

# Summary table
qlr_summary = pd.DataFrame([
    {
        "Asset"            : name,
        "Max. F-Statistik" : r["Max. F-Statistik"],
        "Krit. Wert 10%"   : r["Krit. Wert 10%"],
        "Strukturbruch"    : r["Strukturbruch"],
        "Geschätzter T*"   : r["Geschätzter T*"],
    }
    for name, r in qlr_results.items()
])
print(qlr_summary.to_string(index=False))\
"""))

new_cells.append(code("""\
# --- Plot QLR F-statistics per asset ---
C_GOLD, C_BTC, C_EUR = "#f59e0b", "#f97316", "#3b82f6"

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

for ax, (name, color) in zip(
    axes,
    [("Gold", C_GOLD), ("BTC", C_BTC), ("EUR/USD", C_EUR)],
):
    r = qlr_results[name]
    ax.plot(r["breakpoints"], r["f_stats"], color=color, linewidth=0.9)
    ax.axhline(r["Krit. Wert 10%"], color="red", linewidth=1.2,
               linestyle="--", label=f"Krit. Wert 10% = {r['Krit. Wert 10%']}")
    ax.axvline(r["Geschätzter T*"], color="darkgray", linewidth=1.0,
               linestyle=":", label=f"T* = {r['Geschätzter T*']}")
    ax.set_ylabel(f"F-Statistik ({name})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

axes[0].set_title("QLR-Test F-Statistiken — Strukturbruchsuche", fontsize=13)
plt.tight_layout()
plt.show()\
"""))

new_cells.append(md("""\
**Beobachtung (QLR-Test):**

_(Hier kommt die Interpretation rein — bei welchem Asset wurde ein \
signifikanter Strukturbruch gefunden? Welchem Datum entspricht $T^*$ \
ungefähr? Welche Konsequenzen hat das für die VAR-Schätzung?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 6 — Lag-Selektion
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 6: Lag-Selektion

Die Lag-Länge $p$ des VAR-Modells bestimmt, wie viele vergangene Perioden \
berücksichtigt werden. Zu wenige Lags → Autokorrelation in den Residuen. \
Zu viele Lags → Überanpassung.

### Informationskriterien für VAR

| Kriterium | Formel | Strafterm |
|-----------|--------|-----------|
| **AIC** | $-2 \\ln L + 2k$ | Mild |
| **BIC** | $-2 \\ln L + k \\ln n$ | Stark (bevorzugt für Parsimonie) |
| **HQIC** | $-2 \\ln L + 2k \\ln(\\ln n)$ | Mittel |

Niedrigerer Wert = besser. Wir wählen den BIC-Sieger als Hauptmodell \
und testen zusätzlich VAR(3) als Professorstandard.\
"""))

new_cells.append(code("""\
from statsmodels.tsa.vector_ar.var_model import VAR

# Lag order selection
model    = VAR(df)
lag_sel  = model.select_order(maxlags=10)
p_auto   = int(lag_sel.bic)       # BIC winner

print(lag_sel.summary())
print(f"\\nEmpfohlene Lag-Länge (BIC): p = {p_auto}")\
"""))

new_cells.append(code("""\
# Clean IC table with highlighted winners
ics_dict = lag_sel.ics
ic_rows = []
for lag in range(11):
    aic_val  = float(ics_dict["aic"][lag])  if lag < len(ics_dict["aic"])  else float("nan")
    bic_val  = float(ics_dict["bic"][lag])  if lag < len(ics_dict["bic"])  else float("nan")
    hqic_val = float(ics_dict["hqic"][lag]) if lag < len(ics_dict["hqic"]) else float("nan")
    ic_rows.append({
        "Lag p": lag,
        "AIC"  : round(aic_val,  3),
        "BIC"  : round(bic_val,  3),
        "HQIC" : round(hqic_val, 3),
    })

ic_df = pd.DataFrame(ic_rows)

# Mark winners
for crit in ["AIC", "BIC", "HQIC"]:
    best = ic_df[crit].idxmin()
    ic_df.loc[best, crit] = f"★ {ic_df.loc[best, crit]}"

ic_df\
"""))

new_cells.append(md("""\
**Beobachtung (Lag-Selektion):**

_(Hier kommt die Interpretation rein — welchen Lag empfiehlt BIC? \
Stimmen AIC/BIC/HQIC überein oder widersprechen sie sich? \
Was bedeutet p = 1 oder p = 3 inhaltlich?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 7 — VAR(p_auto) fitten
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 7: VAR(p_auto) fitten und analysieren\
"""))

new_cells.append(code("""\
import warnings
warnings.filterwarnings("ignore")

# Fit VAR with BIC-selected lag
var_auto = VAR(df).fit(p_auto)
print(var_auto.summary())\
"""))

new_cells.append(code("""\
# Residual diagnostics — Durbin-Watson and Portmanteau test
from statsmodels.stats.stattools import durbin_watson

dw_stats = durbin_watson(var_auto.resid)
port_result = var_auto.test_whiteness(nlags=10)

diag_rows = []
for col, dw in zip(df.columns, dw_stats):
    diag_rows.append({
        "Gleichung"  : col.replace("_", "/"),
        "Durbin-Watson": round(dw, 4),
        "DW OK (≈2)" : "✓" if 1.5 < dw < 2.5 else "⚠",
    })

dw_df = pd.DataFrame(diag_rows)
print("=== Durbin-Watson-Statistik ===")
print(dw_df.to_string(index=False))
print()
print(f"=== Portmanteau-Test (multivariate Ljung-Box, Lag 10) ===")
print(f"  Teststatistik : {port_result.test_statistic:.4f}")
print(f"  p-Wert        : {port_result.pvalue:.4f}")
print(f"  Ergebnis      : {'Keine Autokorrelation ✓' if port_result.pvalue > 0.05 else 'Autokorrelation ✗'}")\
"""))

new_cells.append(md("""\
**Beobachtung (VAR(p_auto)):**

_(Hier kommt die Interpretation rein — welche Koeffizienten sind signifikant? \
Bestehen die Residuen den Whiteness-Test? Gibt es Hinweise auf \
Fehlspezifikation?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 8 — VAR(3) fitten und vergleichen
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 8: VAR(3) fitten und vergleichen

Wir fitten zusätzlich **VAR(3)** als Referenzmodell (Standard in der \
Literatur für Tagesdaten) und vergleichen es mit dem BIC-optimierten Modell.\
"""))

new_cells.append(code("""\
# Fit VAR(3) explicitly
var3 = VAR(df).fit(3)
print(var3.summary())\
"""))

new_cells.append(code("""\
# Side-by-side comparison
comparison = pd.DataFrame({
    "Kriterium"    : ["AIC", "BIC", "HQIC", "Log-Likelihood", "Lag p"],
    f"VAR({p_auto})": [
        round(var_auto.aic, 2), round(var_auto.bic, 2),
        round(var_auto.hqic, 2), round(var_auto.llf, 2), p_auto,
    ],
    "VAR(3)"       : [
        round(var3.aic, 2), round(var3.bic, 2),
        round(var3.hqic, 2), round(var3.llf, 2), 3,
    ],
})
comparison\
"""))

new_cells.append(md("""\
**Beobachtung (Modellvergleich VAR(p_auto) vs. VAR(3)):**

_(Hier kommt die Interpretation rein — welches Modell schneidet besser ab? \
Was bedeutet die BIC-Differenz inhaltlich?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 9 — Granger Causality
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 9: Granger-Kausalitätstests

### Konzept

**Granger-Kausalität** testet: „Verbessert die Vergangenheit von Asset X \
die Vorhersage von Asset Y, wenn die eigene Vergangenheit von Y bereits \
berücksichtigt wird?"

- H₀: X hat **keine** Vorhersagekraft für Y (alle Kreuz-Lags = 0)
- H₁: X **Granger-verursacht** Y
- Teststatistik: Wald-Test, signifikant bei p < 0,05

Granger-Kausalität ist kein Beweis für ökonomische Kausalität — sie zeigt \
nur **lineare Vorhersagebeziehungen**.\
"""))

new_cells.append(code("""\
from src.models.var_model import granger_causality_test

gc_results = granger_causality_test(df, lag=3)
gc_df = gc_results["tabelle"]

print("=== Granger-Kausalitätstests (VAR Lag = 3) ===")
print(gc_df[["Beziehung", "F-Statistik", "p-Wert", "Granger-kausal"]].to_string(index=False))\
"""))

new_cells.append(code("""\
# --- Causality diagram (matplotlib, no external dependencies) ---
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Granger-Kausalitäts-Diagramm (VAR(3), p < 0,05)",
             fontsize=13, pad=12)

# Node positions
node_pos = {"Gold": (5, 8.5), "BTC": (1.5, 2.5), "EUR_USD": (8.5, 2.5)}
node_colors = {"Gold": C_GOLD, "BTC": C_BTC, "EUR_USD": C_EUR}
node_labels = {"Gold": "Gold", "BTC": "BTC", "EUR_USD": "EUR/USD"}

# Draw nodes
for name, (x, y) in node_pos.items():
    circle = plt.Circle((x, y), 1.1, color=node_colors[name], zorder=3, alpha=0.9)
    ax.add_patch(circle)
    ax.text(x, y, node_labels[name], ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=4)

# Draw edges
col_map = {"Gold": "Gold", "BTC": "BTC", "EUR/USD": "EUR_USD"}

for _, row in gc_df.iterrows():
    parts = row["Beziehung"].split(" → ")
    cause_key = col_map.get(parts[0], parts[0])
    effect_key = col_map.get(parts[1], parts[1])
    sig = row["Granger-kausal"].startswith("✅")

    x1, y1 = node_pos[cause_key]
    x2, y2 = node_pos[effect_key]
    color = "#16a34a" if sig else "#9ca3af"
    lw    = 2.5 if sig else 1.2
    ls    = "-" if sig else "--"
    alpha = 0.85 if sig else 0.4

    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=ls, alpha=alpha,
            connectionstyle="arc3,rad=0.12",
        ),
    )

    # p-value label on significant arrows
    if sig:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my + 0.15, f"p={row['p-Wert']:.3f}",
                fontsize=8, color=color)

legend_handles = [
    mpatches.Patch(color="#16a34a", label="Signifikant (p<0,05)"),
    mpatches.Patch(color="#9ca3af", label="Nicht signifikant"),
]
ax.legend(handles=legend_handles, loc="lower center", fontsize=10)
plt.tight_layout()
plt.show()\
"""))

new_cells.append(md("""\
**Beobachtung (Granger-Kausalität):**

_(Hier kommt die Interpretation rein — welche Richtungen sind signifikant? \
Was bedeutet das ökonomisch? Reagiert Gold auf BTC-Schocks oder umgekehrt?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 10 — Impulse Response Functions
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 10: Impuls-Antwort-Funktionen (IRF)

### Konzept

Die **Impulse Response Function (IRF)** zeigt: Wie reagiert Asset Y \
auf einen 1-Standardabweichungs-Schock in Asset X, über die nächsten \
h Perioden?

- Identifikation via **Cholesky-Zerlegung**: erfordert eine Annahme \
  über die Kausalreihenfolge
- Wir verwenden: **[Gold, BTC, EUR/USD]** — Gold als exogensten Markt \
  (Rohstoff, historisch weniger reaktiv auf digitale Assets)
- Orthogonale IRF (orth=True): Schocks sind unkorreliert nach Cholesky\
"""))

new_cells.append(code("""\
import matplotlib.pyplot as plt

# Orthogonalised IRF using VAR(3)
irf = var3.irf(periods=20)
irf.plot(orth=True, figsize=(12, 10))
plt.suptitle("Impulse Response Functions (orthogonal) — VAR(3)", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()\
"""))

new_cells.append(code("""\
# Cumulative IRF
irf.plot_cum_effects(orth=True, figsize=(12, 10))
plt.suptitle("Kumulative IRF (orthogonal) — VAR(3)", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()\
"""))

new_cells.append(md("""\
**Beobachtung (IRF):**

_(Hier kommt die Interpretation rein — auf welche Schocks reagieren die \
anderen Assets? Wie schnell klingt die Reaktion ab? \
Gibt es anhaltende Effekte, die auf Kointegration hindeuten?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Section 11 — FEVD
# ════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---
## Schritt 11: Forecast Error Variance Decomposition (FEVD)

### Konzept

Die FEVD beantwortet: **Wie viel Prozent des Prognose-Fehlers von Asset Y \
nach h Perioden sind auf Schocks in Asset X zurückzuführen?**

- Ergänzt den Granger-Test: Granger sagt *ob*, FEVD sagt *wie viel*
- Horizon h = 1: nur eigene Schocks relevant (per Definition)
- Horizon h → ∞: dominante Transmissionskanäle erkennbar\
"""))

new_cells.append(code("""\
# FEVD for VAR(3)
fevd = var3.fevd(periods=20)
fevd.plot(figsize=(12, 8))
plt.suptitle("Forecast Error Variance Decomposition — VAR(3)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()\
"""))

new_cells.append(code("""\
# FEVD table at h = 10
fevd_at_10 = pd.DataFrame(
    fevd.decomp[9],          # period 10 = index 9 (0-based)
    index=df.columns,
    columns=df.columns,
).round(4)

print("=== FEVD bei h = 10 ===")
print("(Zeilen = erklärtes Asset, Spalten = erklärende Schock-Quelle)")
print()
print(fevd_at_10.to_string())\
"""))

new_cells.append(md("""\
**Beobachtung (FEVD):**

_(Hier kommt die Interpretation rein — wie viel Prozent der Goldrenditen \
werden durch BTC-Schocks erklärt? Dominiert der eigene Schock, oder \
spielen Kreuz-Asset-Schocks eine relevante Rolle?)_\
"""))

# ════════════════════════════════════════════════════════════════════
# Write Part 2
# ════════════════════════════════════════════════════════════════════
nb.cells.extend(new_cells)

with open(NB, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Part 2 saved — added {len(new_cells)} cells")
print(f"Total cells: {len(nb.cells)}")
