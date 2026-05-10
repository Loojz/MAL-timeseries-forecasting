# src/models/arima_model.py
# Vollständige Box-Jenkins Methode – alle 7 Schritte
# Ergänzt um Erkenntnisse aus dem Gold-Notebook (Luis):
# - Chow-Test korrekt auf differenzierter Reihe
# - Explizite Kandidatenmodelle + Random Walk Benchmark
# - ADF/KPSS Diskrepanz-Erklärung (Heteroskedastizität)
# - 5-Check Residualdiagnose inkl. Hinweis Fat Tails
# - Beide Modelle (Hauptmodell + Benchmark) parallel

import warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")


# ── Chow-Test auf DIFFERENZIERTER Reihe ──────────────────────────────────────

def chow_test_auf_diff(series_diff: pd.Series, break_date: str) -> dict:
    """
    Chow-Test auf der DIFFERENZIERTEN Reihe (akademisch korrekt, wie im Notebook).
    Testet ob sich der datengenerierende Prozess (nicht Preisniveau) vor/nach T* unterscheidet.

    H0: kein Strukturbruch im Prozess (beta2 = 0)
    H1: Strukturbruch vorhanden
    """
    import statsmodels.api as sm
    from scipy import stats

    y_full = series_diff.dropna().values
    n, k   = len(y_full), 1

    X_const  = sm.add_constant(np.ones(n))
    rss_full = sm.OLS(y_full, X_const).fit().ssr

    try:
        y_pre  = series_diff.loc[:break_date].dropna().values
        y_post = series_diff.loc[break_date:].dropna().values
    except Exception:
        return {"fehler": f"Ungültiges Datum: {break_date}"}

    if len(y_pre) < 10 or len(y_post) < 10:
        return {"fehler": "Zu wenige Beobachtungen in einer Teilstichprobe"}

    rss_pre  = sm.OLS(y_pre,  sm.add_constant(np.ones(len(y_pre)))).fit().ssr
    rss_post = sm.OLS(y_post, sm.add_constant(np.ones(len(y_post)))).fit().ssr
    rss_unr  = rss_pre + rss_post

    chow_f = ((rss_full - rss_unr) / k) / (rss_unr / (n - 2 * k))
    p_val  = stats.f.sf(chow_f, k, n - 2 * k)

    return {
        "Methode":        "Chow-Test (auf differenzierter Reihe)",
        "Bruchpunkt T*":  break_date,
        "N gesamt":       n,
        "N Pre":          len(y_pre),
        "N Post":         len(y_post),
        "F-Statistik":    round(chow_f, 4),
        "p-Wert":         round(p_val, 4),
        "Strukturbruch":  "✅ Ja (p<0.05)" if p_val < 0.05 else "❌ Nein (p≥0.05)",
        "Interpretation": (
            f"H0 abgelehnt: Prozess hat sich bei T*={break_date} verändert → Subsample empfohlen"
            if p_val < 0.05 else
            f"H0 NICHT abgelehnt (F={chow_f:.2f}, p={p_val:.3f})\n"
            f"→ Kein sign. Bruch im Prozess → vollständige Zeitreihe verwenden\n"
            f"→ Mehr Daten = stabilere Parameterschätzung"
        ),
    }


def qlr_test(series: pd.Series, trim: float = 0.15) -> dict:
    """QLR-Test für unbekannten Bruchpunkt (Andrews 1993)."""
    import statsmodels.api as sm
    from scipy import stats

    y = series.dropna().values
    n = len(y)
    start, end = int(n * trim), int(n * (1 - trim))
    f_stats, breakpoints = [], list(range(start, end))

    for t in breakpoints:
        try:
            y_dep    = y[1:]
            nn       = len(y_dep)
            rss_full = sm.OLS(y_dep, sm.add_constant(np.ones(nn))).fit().ssr
            y_pre    = y_dep[:t-1]
            y_post   = y_dep[t-1:]
            rss_unr  = (sm.OLS(y_pre,  sm.add_constant(np.ones(len(y_pre)))).fit().ssr
                        if len(y_pre) > 2 else rss_full) + \
                       (sm.OLS(y_post, sm.add_constant(np.ones(len(y_post)))).fit().ssr
                        if len(y_post) > 2 else rss_full)
            f_stat   = ((rss_full - rss_unr) / 1) / (rss_unr / (nn - 2))
            f_stats.append(max(0, f_stat))
        except Exception:
            f_stats.append(0.0)

    if not f_stats:
        return {"fehler": "QLR-Test fehlgeschlagen"}

    f_max, krit_10 = max(f_stats), 7.17
    t_star = breakpoints[np.argmax(f_stats)]
    return {
        "Methode":          "QLR-Test (T* unbekannt)",
        "Max. F-Statistik": round(f_max, 4),
        "Geschätzter T*":   t_star,
        "Krit. Wert 10%":   krit_10,
        "Strukturbruch":    "✅ Ja" if f_max > krit_10 else "❌ Nein",
        "Interpretation":   (
            f"Bruch bei Index t={t_star} (F={f_max:.2f} > {krit_10})"
            if f_max > krit_10 else
            f"Kein sign. Bruch (F={f_max:.2f} ≤ {krit_10})"
        ),
        "f_stats":     f_stats,
        "breakpoints": breakpoints,
    }


# ── Kandidatenmodelle (wie im Notebook definiert) ─────────────────────────────

KANDIDATEN = [
    ((0, 1, 0), "Random Walk",   "Benchmark – Markteffizienz-Hypothese (EMH)"),
    ((1, 1, 0), "ARIMA(1,1,0)",  "AR(1) – gestriger Tag beeinflusst heute"),
    ((0, 1, 1), "ARIMA(0,1,1)",  "MA(1) – gestriger Schock wirkt nach"),
    ((1, 1, 1), "ARIMA(1,1,1)",  "ARMA(1,1) – kombiniert AR und MA"),
    ((2, 1, 2), "ARIMA(2,1,2)",  "Komplex – nur sinnvoll bei mehreren sign. ACF/PACF-Lags"),
]


def vergleiche_kandidaten(series: pd.Series, train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Vergleicht Kandidatenmodelle wie im Gold-Notebook."""
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rows = []
    for order, name, beschr in KANDIDATEN:
        try:
            mod  = ARIMA(train, order=order).fit()
            fc   = mod.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, fc))
            mae  = mean_absolute_error(test, fc)
            sig  = all(abs(t) > 1.96 for t in mod.tvalues if not np.isnan(t))
            rows.append({
                "Modell":        name,
                "Order":         str(order),
                "AIC":           round(mod.aic, 2),
                "BIC":           round(mod.bic, 2),
                "RMSE (Test)":   round(rmse, 6),
                "MAE (Test)":    round(mae, 6),
                "Koeff. sign.":  "✅" if sig else "⚠️ Überanpassung prüfen",
                "Beschreibung":  beschr,
            })
        except Exception as e:
            rows.append({"Modell": name, "Order": str(order), "Beschreibung": f"Fehler: {e}"})
    return pd.DataFrame(rows).sort_values("BIC", na_position="last").reset_index(drop=True)


# ── Residualdiagnose ──────────────────────────────────────────────────────────

def residual_diagnostics_dict(model) -> dict:
    """5-Check Residualdiagnose (wie im Notebook)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera
    res = model.resid
    lb10 = acorr_ljungbox(res, lags=[10], return_df=True)
    lb20 = acorr_ljungbox(res, lags=[20], return_df=True)
    jb_stat, jb_p = jarque_bera(res)
    return {
        "residuen":       res,
        "mean":           round(float(res.mean()), 6),
        "std":            round(float(res.std()), 4),
        "skew":           round(float(res.skew()), 4),
        "kurtosis":       round(float(res.kurtosis()), 4),
        "lb_p_lag10":     round(float(lb10["lb_pvalue"].iloc[0]), 4),
        "lb_p_lag20":     round(float(lb20["lb_pvalue"].iloc[0]), 4),
        "jb_stat":        round(jb_stat, 4),
        "jb_p":           round(jb_p, 4),
        "keine_autokorr": float(lb10["lb_pvalue"].iloc[0]) > 0.05,
        "normalverteilt": jb_p > 0.05,
        "interpretation": (
            f"Ljung-Box Lag10 p={lb10['lb_pvalue'].iloc[0]:.4f}: "
            f"{'Keine Autokorr. ✅' if float(lb10['lb_pvalue'].iloc[0])>0.05 else 'Autokorr. ❌'}\n"
            f"Jarque-Bera p={jb_p:.4f}: "
            f"{'Normalverteilt ✅' if jb_p>0.05 else 'Nicht normalverteilt ❌ (bei Finanzdaten normal – Fat Tails)'}"
        ),
    }


# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────

def walk_forward_cv(series: pd.Series, order: tuple, n_splits: int = 5) -> dict:
    """Zeitreihen-konforme CV mit expandierendem Fenster."""
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    n, fold = len(series), len(series) // (n_splits + 1)
    rmse_l, mae_l = [], []
    for i in range(n_splits):
        train = series.iloc[:fold*(i+1)]
        test  = series.iloc[fold*(i+1):min(fold*(i+2), n)]
        if not len(test):
            continue
        try:
            mod = ARIMA(train, order=order).fit()
            fc  = mod.forecast(steps=len(test))
            rmse_l.append(np.sqrt(mean_squared_error(test, fc)))
            mae_l.append(mean_absolute_error(test, fc))
        except Exception:
            continue
    if not rmse_l:
        return {"fehler": "CV fehlgeschlagen"}
    return {
        "n_splits":   n_splits,
        "RMSE_folds": [round(r, 6) for r in rmse_l],
        "MAE_folds":  [round(m, 6) for m in mae_l],
        "RMSE_mean":  round(np.mean(rmse_l), 6),
        "RMSE_std":   round(np.std(rmse_l), 6),
        "MAE_mean":   round(np.mean(mae_l), 6),
    }


# ── Vollständige Box-Jenkins Pipeline ─────────────────────────────────────────

def box_jenkins_pipeline(
    series:         pd.Series,
    asset_name:     str,
    forecast_steps: int   = 10,
    train_ratio:    float = 0.70,
    max_p:          int   = 3,
    max_q:          int   = 3,
) -> dict:
    """
    Vollständige Box-Jenkins Methode (alle 7 Schritte + Erweiterungen).
    Basiert auf Gold-Notebook Methodik von Luis.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from src.utils.data import adf_test, kpss_test, train_test_split_ts, berechne_metriken

    results      = {"asset": asset_name, "series_original": series.copy()}
    series_clean = series.dropna()

    # ── Schritt 1: Integration ────────────────────────────────────────────────
    log_ret  = np.log(series_clean / series_clean.shift(1)).dropna()
    diff1    = series_clean.diff().dropna()
    adf_niv  = adf_test(series_clean, f"{asset_name} – Preisniveau")
    adf_ret  = adf_test(log_ret,      f"{asset_name} – Log-Returns")
    adf_d1   = adf_test(diff1,        f"{asset_name} – Erste Differenz")
    kpss_ret = kpss_test(log_ret,     f"{asset_name} – Log-Returns (KPSS)")

    d = 0
    s = series_clean.copy()
    for i in range(3):
        _, p, *_ = adfuller(s.dropna(), autolag="AIC")
        if p < 0.05:
            d = i; break
        s = s.diff().dropna(); d = i + 1

    # ADF/KPSS Diskrepanz erklären (Notebook-Insight)
    diskrepanz = ""
    if (adf_ret.get("Stationär (p < 0.05)") == "✅ Ja" and
            kpss_ret.get("Stationär (p > 0.05)") == "❌ Nein"):
        diskrepanz = (
            "⚠️ ADF und KPSS widersprechen sich – bei Finanzdaten nicht ungewöhnlich.\n"
            "ADF bestätigt: kein Unit Root → stationär ✅\n"
            "KPSS reagiert auf zeitlich variierende Varianz (Heteroskedastizität).\n"
            "Ruhige Phasen wechseln mit turbulenten → ARCH-Effekte.\n"
            "→ Für ARIMA reicht ADF: d=1 ausreichend.\n"
            "→ Heteroskedastizität → GARCH (außerhalb ARIMA-Rahmen)."
        )

    results["schritt1_integration"] = {
        "d": d, "adf_niveau": adf_niv, "adf_returns": adf_ret,
        "adf_diff1": adf_d1, "kpss_returns": kpss_ret,
        "diskrepanz": diskrepanz,
        "interpretation": f"Zeitreihe ist I({d}). Log-Returns → I(0).",
    }

    # ── Schritt 2: Transformation ─────────────────────────────────────────────
    series_stat = log_ret.copy()
    results["schritt2_transformation"] = {
        "series_stationaer": series_stat,
        "methode": "Log-Returns: r_t = ln(P_t / P_{t-1})\n• I(1)→I(0) • Symmetrisch • Varianzstabilisierend",
        "d_verwendet": 1,
    }

    # ── Train/Test Split ──────────────────────────────────────────────────────
    train, test = train_test_split_ts(series_stat, train_ratio)
    results["train_test"] = {
        "n_train": len(train), "n_test": len(test),
        "ratio": f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}",
    }

    # ── Strukturbrüche auf differenzierter Reihe (Notebook-Methodik) ──────────
    chow_res = {}
    for label, datum in [("COVID-19 (2020-03-15)", "2020-03-15"),
                          ("Ukraine-Invasion (2022-02-24)", "2022-02-24")]:
        try:
            chow_res[label] = chow_test_auf_diff(series_stat, datum)
        except Exception as e:
            chow_res[label] = {"fehler": str(e)}
    results["strukturbrueche_diff"] = chow_res

    try:
        results["qlr"] = qlr_test(series_clean)
    except Exception as e:
        results["qlr"] = {"fehler": str(e)}

    # ── Schritt 3: ACF & PACF ─────────────────────────────────────────────────
    nlags     = min(40, len(series_stat) // 4)
    acf_vals  = acf(series_stat, nlags=nlags, fft=True)
    pacf_vals = pacf(series_stat, nlags=nlags)
    conf_int  = 1.96 / np.sqrt(len(series_stat))
    results["schritt3_acf_pacf"] = {
        "acf": acf_vals, "pacf": pacf_vals,
        "konfidenzband": conf_int, "nlags": nlags,
        "interpretation": (
            f"Band: ±{conf_int:.4f} | ACF→MA(q) | PACF→AR(p)\n"
            f"Bei effizienten Märkten: kaum sign. Lags → Random Walk"
        ),
    }

    # ── Schritt 4: Kandidaten + Grid Search ───────────────────────────────────
    df_kand = vergleiche_kandidaten(series_clean, train, test)
    grid_rows, beste_aic, bestes_order, bestes_mod = [], np.inf, (0,1,1), None

    for p_ord, q_ord in product(range(max_p+1), range(max_q+1)):
        if p_ord == 0 and q_ord == 0:
            continue
        try:
            mod = ARIMA(train, order=(p_ord, 0, q_ord)).fit()
            fc  = mod.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, fc))
            grid_rows.append({
                "ARIMA": f"({p_ord},0,{q_ord})",
                "AIC": round(mod.aic,2), "BIC": round(mod.bic,2),
                "RMSE (Test)": round(rmse,6),
            })
            if mod.aic < beste_aic:
                beste_aic, bestes_order, bestes_mod = mod.aic, (p_ord,0,q_ord), mod
        except Exception:
            continue

    df_grid = pd.DataFrame(grid_rows).sort_values("AIC").reset_index(drop=True)
    results["schritt4_selektion"] = {
        "kandidaten_tabelle": df_kand,
        "grid_tabelle":       df_grid,
        "bestes_order":       bestes_order,
        "bestes_aic":         round(beste_aic, 2),
        "bestes_modell":      bestes_mod,
        "interpretation": (
            f"BIC-Sieger: {df_kand.iloc[0]['Modell']} (BIC={df_kand.iloc[0]['BIC']:.2f})\n"
            f"Grid AIC-Sieger: ARIMA{bestes_order}\n"
            f"⚠️ Koeff. nicht sign. → Überanpassung (wie ARIMA(2,1,2) im Gold-Notebook)"
        ),
    }

    if bestes_mod is None:
        results["fehler"] = "Kein Modell gefittet"
        return results

    # ── CV ────────────────────────────────────────────────────────────────────
    results["cross_validation"] = walk_forward_cv(series_stat, bestes_order)

    # ── Schritt 5: Residualdiagnose (5-Check) ─────────────────────────────────
    diag = residual_diagnostics_dict(bestes_mod)
    try:
        rw_m_obj   = ARIMA(train, order=(0,0,0)).fit()
        diag_bench = residual_diagnostics_dict(rw_m_obj)
    except Exception:
        diag_bench = None

    results["schritt5_residuen"] = {
        "hauptmodell":  diag,
        "benchmark":    diag_bench,
        "statistiken": {
            "Mittelwert (≈0)":          diag["mean"],
            "Standardabweichung":        diag["std"],
            "Schiefe":                   diag["skew"],
            "Kurtosis":                  diag["kurtosis"],
            "Ljung-Box p (Lag 10)":      diag["lb_p_lag10"],
            "Ljung-Box p (Lag 20)":      diag["lb_p_lag20"],
            "Jarque-Bera p":             diag["jb_p"],
        },
        "hinweis": (
            "Jarque-Bera oft p<0.05 bei Finanzdaten → normal! Fat Tails sind bekannt.\n"
            "Entscheidend: Ljung-Box (Autokorrelation in Residuen).\n"
            "Volatilitäts-Cluster → GARCH wäre nächster Schritt."
        ),
        "interpretation": diag["interpretation"],
    }

    # ── Schritt 6: t-Statistiken ──────────────────────────────────────────────
    koeff_df = pd.DataFrame({
        "Koeffizient":  bestes_mod.params.index,
        "Schätzwert":   bestes_mod.params.values.round(6),
        "Std. Fehler":  bestes_mod.bse.values.round(6),
        "t-Statistik":  bestes_mod.tvalues.values.round(4),
        "p-Wert":       bestes_mod.pvalues.values.round(4),
        "Signifikant":  ["✅" if abs(t)>1.96 else "❌" for t in bestes_mod.tvalues.values],
    })
    results["schritt6_koeffizienten"] = {
        "tabelle": koeff_df,
        "interpretation": "|t|>1.96 → signifikant auf 5%-Niveau | nicht sign. → Modell vereinfachen",
    }

    # ── Schritt 7: Prognose ────────────────────────────────────────────────────
    try:
        full_mod = ARIMA(series_stat, order=bestes_order).fit()
        fc       = full_mod.get_forecast(steps=forecast_steps)
        ci       = fc.conf_int(alpha=0.05)
        results["schritt7_prognose"] = {
            "prognose_mean":   fc.predicted_mean,
            "konfidenz_lower": ci.iloc[:, 0],
            "konfidenz_upper": ci.iloc[:, 1],
            "prognose_df":     fc.summary_frame(alpha=0.05),
            "schritte":        forecast_steps,
            "formel":          "ŷ_{t+h|t} ± 1.96·σ̂_h",
            "interpretation":  (
                f"ARIMA{bestes_order}: {forecast_steps} Perioden ahead\n"
                f"KI wächst mit h → zunehmende Unsicherheit"
            ),
        }
    except Exception as e:
        results["schritt7_prognose"] = {"fehler": str(e)}

    # ── Benchmark: Random Walk (Markteffizienz) ───────────────────────────────
    rw_fc      = np.full(len(test), train.iloc[-1])
    rw_met     = berechne_metriken(test, rw_fc, "Random Walk")
    arima_fc   = bestes_mod.forecast(steps=len(test))
    arima_met  = berechne_metriken(test, arima_fc, f"ARIMA{bestes_order}")
    results["benchmark_vergleich"] = {
        "random_walk":       rw_met,
        "arima":             arima_met,
        "arima_besser_rmse": arima_met.get("RMSE",999) < rw_met.get("RMSE",999),
        "interpretation": (
            f"ARIMA{bestes_order} vs. Random Walk:\n"
            f"RMSE ARIMA: {arima_met.get('RMSE','–')} | RMSE RW: {rw_met.get('RMSE','–')}\n"
            f"EUR/USD-Teammitglied: ARIMA(0,1,0) gewinnt → klassischer Random Walk ✅"
        ),
    }

    return results
