# src/models/var_model.py
# Teil 3: Multivariate Zeitreihenanalyse
# VAR-Modelle, Granger Causality Test, Prognosen, State Space Models

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def bereite_var_daten_vor(dfs: dict) -> pd.DataFrame:
    """
    Bereitet alle Zeitreihen für VAR-Analyse vor.
    - Log-Returns (stationäre Transformation)
    - Gemeinsamer Datums-Index (inner join)
    """
    returns = {}
    for name, df in dfs.items():
        lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        lr.index = df["Date"].iloc[1:].values
        lr.name  = name
        returns[name] = lr

    df_combined = pd.DataFrame(returns).dropna()
    df_combined.index = pd.to_datetime(df_combined.index)
    return df_combined


def var_lag_selektion(df_returns: pd.DataFrame, max_lags: int = 10) -> dict:
    """
    Bestimmt optimale Lag-Länge p via AIC, BIC, HQ (Folie 26).
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    model    = VAR(df_returns)
    results  = model.select_order(maxlags=max_lags)
    aic_lag  = results.aic
    bic_lag  = results.bic
    hqic_lag = results.hqic

    # Tabelle aller Informationskriterien
    # results.ics ist ein dict mit Listen (index = lag 0..max_lags)
    ics = results.ics
    aic_list  = ics.get("aic",  []) if isinstance(ics, dict) else []
    bic_list  = ics.get("bic",  []) if isinstance(ics, dict) else []
    hqic_list = ics.get("hqic", []) if isinstance(ics, dict) else []

    def safe_get(lst, i):
        try:
            return round(float(lst[i]), 4)
        except Exception:
            return np.nan

    rows = []
    for lag in range(max_lags + 1):
        rows.append({
            "Lag p": lag,
            "AIC":   safe_get(aic_list,  lag),
            "BIC":   safe_get(bic_list,  lag),
            "HQIC":  safe_get(hqic_list, lag),
        })

    return {
        "aic_lag":    aic_lag,
        "bic_lag":    bic_lag,
        "hqic_lag":   hqic_lag,
        "empfohlen":  bic_lag,  # BIC bevorzugt (Parsimonie)
        "tabelle":    pd.DataFrame(rows),
        "interpretation": (
            f"AIC → p={aic_lag} | BIC → p={bic_lag} | HQIC → p={hqic_lag}\n"
            f"Empfehlung: p={bic_lag} (BIC penalisiert Komplexität stärker)"
        ),
    }


def var_modell_fitten(df_returns: pd.DataFrame, lag: int) -> object:
    """
    Fittet VAR(p) Modell auf Log-Returns (Folie 24-25).
    Alle Zeitreihen müssen stationär sein.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    model  = VAR(df_returns)
    result = model.fit(lag)
    return result


def granger_causality_test(df_returns: pd.DataFrame, lag: int) -> dict:
    """
    Granger Causality Tests für alle Paare (Folie 27-28).
    H0: Variable x hat KEINE Vorhersagekraft für Variable y
    H1: x Granger-verursacht y

    Verwendet Wald-Test: H0: Φ_{1,xy} = Φ_{2,xy} = ... = 0
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    variables = df_returns.columns.tolist()
    results   = {}

    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue
            key = f"{causing} → {caused}"
            try:
                gc_res = grangercausalitytests(
                    df_returns[[caused, causing]], maxlag=lag, verbose=False
                )
                # F-Test p-Wert für optimale Lag-Länge
                p_val  = gc_res[lag][0]["ssr_ftest"][1]
                f_stat = gc_res[lag][0]["ssr_ftest"][0]
                results[key] = {
                    "Beziehung":       key,
                    "F-Statistik":     round(f_stat, 4),
                    "p-Wert":          round(p_val, 4),
                    "Granger-kausal":  "Ja (p<0.05)" if p_val < 0.05 else "Nein",
                    "Interpretation":  (
                        f"{causing} Granger-verursacht {caused}"
                        if p_val < 0.05 else
                        f"{causing} hat keine signifikante Vorhersagekraft für {caused}"
                    ),
                }
            except Exception as e:
                results[key] = {"fehler": str(e)}

    df_gc = pd.DataFrame([
        v for v in results.values() if "fehler" not in v
    ])

    return {
        "details":        results,
        "tabelle":        df_gc,
        "interpretation": "Granger Causality: x → y wenn x bessere Vorhersage von y ermöglicht",
    }


def var_prognose(var_result, steps: int = 10) -> dict:
    """
    Rekursive VAR(p) Prognose (Folie 30).
    ŷ_{t+1} = c + Φ_1 y_t + ... + Φ_p y_{t-p+1}
    """
    fc       = var_result.forecast(var_result.endog[-var_result.k_ar:], steps=steps)
    fc_df    = pd.DataFrame(fc, columns=var_result.names)
    fc_inter = var_result.forecast_interval(
        var_result.endog[-var_result.k_ar:], steps=steps, alpha=0.05
    )
    return {
        "prognose":       fc_df,
        "lower_95":       pd.DataFrame(fc_inter[1], columns=var_result.names),
        "upper_95":       pd.DataFrame(fc_inter[2], columns=var_result.names),
        "schritte":       steps,
        "interpretation": f"Rekursive VAR({var_result.k_ar})-Prognose: {steps} Perioden ahead",
    }


def var_evaluation(df_returns: pd.DataFrame, lag: int,
                   train_ratio: float = 0.70) -> dict:
    """
    Train/Test Evaluation des VAR-Modells.
    Vergleich: VAR(p) vs. univariate Random Walk Benchmark.
    """
    from src.utils.data import berechne_metriken
    split   = int(len(df_returns) * train_ratio)
    train   = df_returns.iloc[:split]
    test    = df_returns.iloc[split:]

    try:
        var_mod = var_modell_fitten(train, lag)
        fc      = var_mod.forecast(train.values[-lag:], steps=len(test))
        fc_df   = pd.DataFrame(fc, columns=df_returns.columns, index=test.index)
    except Exception as e:
        return {"fehler": str(e)}

    metriken = {}
    for col in df_returns.columns:
        arima_m  = berechne_metriken(test[col], fc_df[col], f"VAR({lag})")
        # Random Walk Benchmark
        rw_pred  = np.full(len(test), train[col].iloc[-1])
        rw_m     = berechne_metriken(test[col], rw_pred, "Random Walk")
        metriken[col] = {"VAR": arima_m, "RandomWalk": rw_m}

    return {
        "metriken":       metriken,
        "n_train":        len(train),
        "n_test":         len(test),
        "fc_df":          fc_df,
        "test_df":        test,
    }


def state_space_ets(series: pd.Series, asset_name: str,
                    forecast_steps: int = 10) -> dict:
    """
    State Space Model via ETS (Error-Trend-Seasonal) – Folie 31.
    Observation equation: y_t = A_t x_t + v_t
    State equation:       x_t = Φ x_{t-1} + w_t

    ETS ist eine praktische Implementierung des State Space Frameworks.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from src.utils.data import berechne_metriken, train_test_split_ts

    train, test = train_test_split_ts(series, 0.70)

    try:
        # ETS(A,N,N) – Simple Exponential Smoothing (State Space)
        model = ExponentialSmoothing(
            train,
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

        fc     = model.forecast(steps=len(test))
        fc_fut = model.forecast(steps=forecast_steps)
        metrics = berechne_metriken(test, fc, f"ETS – {asset_name}")

        return {
            "modell":         model,
            "alpha":          round(model.params.get("smoothing_level", np.nan), 4),
            "prognose":       fc_fut,
            "metriken":       metrics,
            "fc_test":        fc,
            "test":           test,
            "interpretation": (
                f"State Space / ETS für {asset_name}\n"
                f"Smoothing-Parameter α={model.params.get('smoothing_level', np.nan):.4f}\n"
                f"Observation eq.: y_t = x_t + v_t | State eq.: x_t = x_{{t-1}} + α·ε_t"
            ),
        }
    except Exception as e:
        return {"fehler": str(e)}
