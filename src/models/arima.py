"""
ARIMA modelling utilities for the MAL Time Series Forecasting project.

All functions are reusable across team members' notebooks (Gold, BTC, EUR/USD).
Import from src.models.arima.
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(
    series: pd.Series,
    order: tuple[int, int, int],
) -> object:
    """
    Fit a non-seasonal ARIMA model using SARIMAX.

    Parameters
    ----------
    series : pd.Series
        The raw (undifferenced) price series. SARIMAX handles differencing
        internally based on the d parameter in `order`.
    order : tuple of (p, d, q)
        The ARIMA order. p = AR lags, d = differencing order, q = MA lags.

    Returns
    -------
    statsmodels SARIMAXResultsWrapper
        Fitted model result. Call .summary() for the full output.

    Examples
    --------
    >>> from src.models.arima import fit_arima
    >>> model = fit_arima(gold["Close"], order=(1, 1, 1))
    >>> print(model.summary())
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series.dropna(),
            order=order,
            trend="n",                     # no deterministic trend term
            enforce_stationarity=False,    # allow fitting even at boundary
            enforce_invertibility=False,   # same for MA roots
        )
        result = model.fit(disp=False)
    return result


def compare_models(
    series: pd.Series,
    candidates: list[tuple[int, int, int]],
    names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fit multiple ARIMA candidates and compare them by AIC and BIC.

    Convergence failures are caught gracefully and marked in the output
    rather than crashing, so the comparison always returns a complete table.

    Parameters
    ----------
    series : pd.Series
        The raw (undifferenced) price series passed to each model.
    candidates : list of (p, d, q) tuples
        ARIMA orders to compare, e.g. [(0,1,0), (1,1,1), (2,1,2)].
    names : list of str, optional
        Human-readable labels for each candidate. If omitted, labels are
        generated automatically as "ARIMA(p,d,q)".

    Returns
    -------
    pd.DataFrame
        One row per candidate, sorted by AIC ascending. Columns:
        model, p, d, q, AIC, BIC, log_likelihood, n_params,
        converged, notes.

    Examples
    --------
    >>> from src.models.arima import compare_models
    >>> results = compare_models(gold["Close"], [(0,1,0), (1,1,0), (0,1,1)])
    """
    if names is None:
        names = [f"ARIMA{order}" for order in candidates]

    rows = []
    for order, name in zip(candidates, names):
        p, d, q = order
        row: dict = {
            "model": name, "p": p, "d": d, "q": q,
            "AIC": None, "BIC": None,
            "log_likelihood": None, "n_params": None,
            "converged": False, "notes": "",
        }
        try:
            result = fit_arima(series, order=order)
            row["AIC"]            = round(result.aic, 2)
            row["BIC"]            = round(result.bic, 2)
            row["log_likelihood"] = round(result.llf, 2)
            row["n_params"]       = int(result.df_model) + 1  # +1 for variance
            row["converged"]      = result.mle_retvals.get("converged", True)
            if not row["converged"]:
                row["notes"] = "did not converge"
        except Exception as exc:  # noqa: BLE001
            row["converged"] = False
            row["notes"]     = f"error: {exc}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by AIC; put failed rows (AIC=None) at the bottom
    df = df.sort_values("AIC", ascending=True, na_position="last").reset_index(drop=True)
    return df


def model_summary_table(fitted_model: object) -> pd.DataFrame:
    """
    Extract a clean coefficient table from a fitted ARIMA / SARIMAX model.

    Parameters
    ----------
    fitted_model : SARIMAXResultsWrapper
        A model already fitted via fit_arima() or directly via SARIMAX.

    Returns
    -------
    pd.DataFrame
        One row per parameter with columns:
        coefficient, std_err, t_statistic, p_value, significant_05.

    Examples
    --------
    >>> from src.models.arima import fit_arima, model_summary_table
    >>> model = fit_arima(gold["Close"], order=(1, 1, 1))
    >>> model_summary_table(model)
    """
    params  = fitted_model.params
    bse     = fitted_model.bse
    tvalues = fitted_model.tvalues
    pvalues = fitted_model.pvalues

    df = pd.DataFrame({
        "coefficient":   params.round(6),
        "std_err":       bse.round(6),
        "t_statistic":   tvalues.round(4),
        "p_value":       pvalues.round(4),
        "significant_05": pvalues < 0.05,
    })
    df.index.name = "parameter"
    return df


def residual_diagnostics(
    fitted_model: object,
    lags: list[int] | None = None,
) -> dict:
    """
    Run standard residual diagnostic tests on a fitted ARIMA model.

    Checks whether the residuals resemble white noise, which is the
    core assumption for a well-specified ARIMA model:
      - No remaining autocorrelation (Ljung-Box test)
      - Approximate normality (Jarque-Bera test; often rejected for financials)
      - Mean close to zero

    Note: For financial return series, a significant Jarque-Bera result
    (heavy tails) is expected and does not invalidate the ARIMA model.
    The absence of autocorrelation (Ljung-Box) is the critical criterion.

    Parameters
    ----------
    fitted_model : SARIMAXResultsWrapper
        A model already fitted via fit_arima().
    lags : list of int, optional
        Lag values for the Ljung-Box test. Defaults to [10, 20].

    Returns
    -------
    dict
        Keys: lb_stat_lag{k}, lb_p_lag{k} for each lag k,
              jb_stat, jb_p, mean, std, skew, kurtosis.

    Examples
    --------
    >>> from src.models.arima import fit_arima, residual_diagnostics
    >>> model = fit_arima(gold["Close"], order=(0, 1, 1))
    >>> diag = residual_diagnostics(model, lags=[10, 20])
    """
    if lags is None:
        lags = [10, 20]

    # Drop the first residual: it is an initialization artifact in SARIMAX
    resid = fitted_model.resid.iloc[1:]

    output: dict = {}

    # ── Ljung-Box test for autocorrelation ────────────────────────────────────
    print("\n" + "=" * 55)
    print("Ljung-Box-Test (H0: keine Autokorrelation)")
    print("=" * 55)
    for lag in lags:
        try:
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            stat = float(lb["lb_stat"].iloc[0])
            pval = float(lb["lb_pvalue"].iloc[0])
            verdict = "nicht signifikant ✓" if pval > 0.05 else "SIGNIFIKANT ✗"
            print(f"  Lag {lag:2d}:  Q = {stat:8.4f},  p = {pval:.4f}  ({verdict})")
            output[f"lb_stat_lag{lag}"] = round(stat, 4)
            output[f"lb_p_lag{lag}"]   = round(pval, 4)
        except Exception as exc:  # noqa: BLE001
            print(f"  Lag {lag}: Fehler — {exc}")
            output[f"lb_stat_lag{lag}"] = None
            output[f"lb_p_lag{lag}"]   = None

    # ── Jarque-Bera normality test ────────────────────────────────────────────
    jb_stat, jb_p = jarque_bera(resid.dropna())
    jb_verdict = "nicht abgelehnt" if jb_p > 0.05 else "ABGELEHNT (Heavy Tails erwartet)"
    print("\n" + "=" * 55)
    print("Jarque-Bera-Test (H0: Normalverteilung)")
    print("=" * 55)
    print(f"  JB-Statistik : {jb_stat:.4f}")
    print(f"  p-Wert       : {jb_p:.4f}  → H0 {jb_verdict}")
    output["jb_stat"] = round(jb_stat, 4)
    output["jb_p"]    = round(jb_p, 4)

    # ── Descriptive statistics ────────────────────────────────────────────────
    r_mean = float(resid.mean())
    r_std  = float(resid.std())
    r_skew = float(stats.skew(resid.dropna()))
    r_kurt = float(stats.kurtosis(resid.dropna()))  # excess kurtosis (Fisher)
    print("\n" + "=" * 55)
    print("Deskriptive Statistiken der Residuen")
    print("=" * 55)
    print(f"  Mittelwert        : {r_mean:10.4f}")
    print(f"  Standardabweichung: {r_std:10.4f}")
    print(f"  Schiefe (Skew)    : {r_skew:10.4f}")
    print(f"  Excess Kurtosis   : {r_kurt:10.4f}  (Normalverteilung = 0)")
    print("=" * 55 + "\n")

    output.update({"mean": round(r_mean, 4), "std": round(r_std, 4),
                   "skew": round(r_skew, 4), "kurtosis": round(r_kurt, 4)})
    return output


def plot_residual_diagnostics(
    fitted_model: object,
    title: str | None = None,
) -> plt.Figure:
    """
    Produce a 2x2 diagnostic plot panel for a fitted ARIMA model.

    The four panels are:
      - Top-left : Residuals over time (line plot)
      - Top-right: Histogram of residuals with normal density overlay
      - Bottom-left : QQ-plot against the normal distribution
      - Bottom-right: ACF of residuals (up to 40 lags)

    Parameters
    ----------
    fitted_model : SARIMAXResultsWrapper
        A model already fitted via fit_arima().
    title : str or None, optional
        Overall figure title displayed above the 2x2 grid.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from src.models.arima import fit_arima, plot_residual_diagnostics
    >>> model = fit_arima(gold["Close"], order=(0, 1, 1))
    >>> plot_residual_diagnostics(model, title="Residuendiagnostik ARIMA(0,1,1)")
    """
    # Drop initialization artifact (first residual)
    resid = fitted_model.resid.iloc[1:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_ts, ax_hist, ax_qq, ax_acf = axes.flatten()

    # ── Top-left: Residuals over time ─────────────────────────────────────────
    ax_ts.plot(resid.index, resid.values, color="#4A7EA8", linewidth=0.7)
    ax_ts.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_ts.set_title("Residuen uber die Zeit", fontsize=12)
    ax_ts.set_xlabel("Datum", fontsize=10)
    ax_ts.set_ylabel("Residuum", fontsize=10)
    ax_ts.grid(True, linestyle="--", alpha=0.4)

    # ── Top-right: Histogram with normal density overlay ──────────────────────
    r_clean = resid.dropna().values
    ax_hist.hist(r_clean, bins=60, density=True, color="#C9A84C",
                 alpha=0.7, edgecolor="white", linewidth=0.4)
    xmin, xmax = ax_hist.get_xlim()
    x_range = np.linspace(xmin, xmax, 300)
    ax_hist.plot(x_range,
                 stats.norm.pdf(x_range, loc=r_clean.mean(), scale=r_clean.std()),
                 color="crimson", linewidth=1.5, label="Normalverteilung")
    ax_hist.set_title("Histogramm der Residuen", fontsize=12)
    ax_hist.set_xlabel("Residuum", fontsize=10)
    ax_hist.set_ylabel("Dichte", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, linestyle="--", alpha=0.4)

    # ── Bottom-left: QQ-plot ──────────────────────────────────────────────────
    stats.probplot(r_clean, dist="norm", plot=ax_qq)
    ax_qq.set_title("QQ-Plot (Normal)", fontsize=12)
    ax_qq.get_lines()[0].set(color="#4A7EA8", markersize=2, alpha=0.5)
    ax_qq.get_lines()[1].set(color="crimson", linewidth=1.5)
    ax_qq.grid(True, linestyle="--", alpha=0.4)

    # ── Bottom-right: ACF of residuals ────────────────────────────────────────
    plot_acf(r_clean, lags=40, ax=ax_acf, zero=False)
    ax_acf.set_title("ACF der Residuen", fontsize=12)
    ax_acf.set_xlabel("Lag", fontsize=10)
    ax_acf.set_ylabel("Korrelation", fontsize=10)
    ax_acf.grid(True, linestyle="--", alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01, fontweight="bold")

    plt.tight_layout()
    plt.show()
    return fig


def make_forecast(
    fitted_model: object,
    steps: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a multi-step-ahead point forecast with confidence intervals.

    Parameters
    ----------
    fitted_model : SARIMAXResultsWrapper
        A model already fitted via fit_arima().
    steps : int, optional
        Number of future periods to forecast. Default is 10.
    alpha : float, optional
        Significance level for the confidence interval. Default is 0.05
        (i.e. 95% confidence interval).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (business days from last training date) with columns:
        forecast, lower_ci, upper_ci, std_err.

    Examples
    --------
    >>> from src.models.arima import fit_arima, make_forecast
    >>> model = fit_arima(gold["Close"], order=(0, 1, 1))
    >>> fc = make_forecast(model, steps=10, alpha=0.05)
    >>> fc.round(2)
    """
    fc_obj   = fitted_model.get_forecast(steps=steps)
    pred     = fc_obj.predicted_mean
    ci       = fc_obj.conf_int(alpha=alpha)
    se       = fc_obj.se_mean

    # Build a business-day DatetimeIndex from the last fitted date
    last_date   = fitted_model.fittedvalues.index[-1]
    future_idx  = pd.bdate_range(start=last_date, periods=steps + 1, freq="B")[1:]

    df = pd.DataFrame({
        "forecast": pred.values,
        "lower_ci": ci.iloc[:, 0].values,
        "upper_ci": ci.iloc[:, 1].values,
        "std_err":  se.values,
    }, index=future_idx)
    return df


def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    title: str | None = None,
    n_history: int = 180,
) -> plt.Figure:
    """
    Plot historical prices and a single-model forecast with confidence band.

    Parameters
    ----------
    history : pd.Series
        Full historical price series with DatetimeIndex.
    forecast_df : pd.DataFrame
        Output of make_forecast() — must have columns forecast, lower_ci, upper_ci.
    title : str or None, optional
        Figure title.
    n_history : int, optional
        Number of historical periods to show before the forecast start. Default 180.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from src.models.arima import make_forecast, plot_forecast
    >>> fc = make_forecast(model, steps=10)
    >>> plot_forecast(gold["Close"], fc, title="10-Tage-Forecast")
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Last n_history observations
    hist_tail = history.iloc[-n_history:]
    ax.plot(hist_tail.index, hist_tail.values,
            color="black", lw=1.0, label="Historische Werte")

    # Point forecast
    ax.plot(forecast_df.index, forecast_df["forecast"],
            color="#5A7FA8", lw=1.8, ls="--", label="Punktforecast")

    # Confidence band
    ax.fill_between(forecast_df.index,
                    forecast_df["lower_ci"], forecast_df["upper_ci"],
                    color="#5A7FA8", alpha=0.25, label="95%-Konfidenzintervall")

    # Vertical line at forecast start
    ax.axvline(forecast_df.index[0], color="gray", lw=0.9, ls=":", alpha=0.8)

    ax.set_xlabel("Datum", fontsize=11)
    ax.set_ylabel("Goldpreis (USD)", fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    plt.show()
    return fig


def plot_forecast_comparison(
    history: pd.Series,
    forecast_dfs_dict: dict[str, pd.DataFrame],
    title: str | None = None,
    n_history: int = 180,
) -> plt.Figure:
    """
    Plot historical prices and multiple model forecasts on the same axes.

    Designed for the Option C narrative: comparing ARIMA(0,1,1) against the
    ARIMA(0,1,0) Random Walk benchmark on the same chart.

    Parameters
    ----------
    history : pd.Series
        Full historical price series with DatetimeIndex.
    forecast_dfs_dict : dict of str -> pd.DataFrame
        Mapping from model label to make_forecast() output.
        E.g. {"ARIMA(0,1,1)": fc1, "ARIMA(0,1,0)": fc2}.
    title : str or None, optional
        Figure title.
    n_history : int, optional
        Number of historical periods to show before forecast start. Default 180.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from src.models.arima import plot_forecast_comparison
    >>> plot_forecast_comparison(gold["Close"],
    ...     {"ARIMA(0,1,1)": fc1, "ARIMA(0,1,0)": fc2},
    ...     title="Forecast-Vergleich")
    """
    colors = ["#5A7FA8", "#D94F3D", "#3DAF6A", "#A85A7F"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical tail
    hist_tail = history.iloc[-n_history:]
    ax.plot(hist_tail.index, hist_tail.values,
            color="black", lw=1.0, label="Historische Werte")

    # One set of bands + line per model
    for (label, fc_df), color in zip(forecast_dfs_dict.items(), colors):
        ax.plot(fc_df.index, fc_df["forecast"],
                color=color, lw=1.8, ls="--", label=f"{label} Punktforecast")
        ax.fill_between(fc_df.index, fc_df["lower_ci"], fc_df["upper_ci"],
                        color=color, alpha=0.15,
                        label=f"{label} 95%-KI")

    # Vertical line at forecast start (use first model's index)
    first_fc = next(iter(forecast_dfs_dict.values()))
    ax.axvline(first_fc.index[0], color="gray", lw=0.9, ls=":", alpha=0.8)

    ax.set_xlabel("Datum", fontsize=11)
    ax.set_ylabel("Goldpreis (USD)", fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    plt.show()
    return fig


def forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray | None = None,
) -> dict:
    """
    Compute standard point-forecast accuracy metrics.

    Parameters
    ----------
    actual : array-like
        True observed values (test set).
    predicted : array-like
        Model point forecasts aligned with `actual`.
    training_series : array-like or None, optional
        Training data used to scale MASE. If provided, the naive benchmark
        is the mean absolute one-step difference on the training series.
        If None, the naive benchmark is computed from `actual` itself
        (less reliable but avoids needing training data).

    Returns
    -------
    dict
        Keys: MAE, RMSE, MAPE, MASE.

    Notes
    -----
    - MAPE is in percent.
    - MASE < 1 means the model beats the naive random-walk baseline.
    - Division-by-zero in MASE (constant training series) is caught and
      returns NaN.

    Examples
    --------
    >>> from src.models.arima import forecast_metrics
    >>> metrics = forecast_metrics(gold_test.values, fc["forecast"].values,
    ...                            training_series=gold_train.values)
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    abs_err = np.abs(errors)

    mae  = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / actual)) * 100)

    # MASE: scale by mean abs one-step difference on training data (naive forecast)
    try:
        if training_series is not None:
            ts = np.asarray(training_series, dtype=float)
            naive_scale = np.mean(np.abs(np.diff(ts)))
        else:
            naive_scale = np.mean(np.abs(np.diff(actual)))
        if naive_scale == 0:
            raise ZeroDivisionError("Constant series — MASE undefined.")
        mase = float(mae / naive_scale)
    except Exception:  # noqa: BLE001
        mase = float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "MASE": round(mase, 4)}


def simulate_random_walk_paths(
    start_value: float,
    sigma: float,
    steps: int,
    n_paths: int = 1000,
    drift: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate random walk paths for Monte-Carlo visualization.

    Each path starts at `start_value` and evolves as:
        y_{t+1} = y_t + drift + ε_t,   ε_t ~ N(0, sigma²)

    With drift=0 (default) this is a pure random walk. The resulting
    "spaghetti" of paths visualises the actual distribution of future
    outcomes, complementing the analytical point forecast and CI.

    Parameters
    ----------
    start_value : float
        Starting value for every path (typically the last observed price).
    sigma : float
        Standard deviation of the per-step innovations.
    steps : int
        Number of time steps to simulate forward.
    n_paths : int, optional
        Number of independent paths to simulate. Default 1000.
    drift : float, optional
        Per-step drift term added to each innovation. Default 0.0.
    seed : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    np.ndarray
        Shape (n_paths, steps + 1). Column 0 is `start_value` for all paths.
        Rows are individual simulated trajectories.

    Examples
    --------
    >>> from src.models.arima import simulate_random_walk_paths
    >>> paths = simulate_random_walk_paths(3200.0, sigma=17.0, steps=10)
    >>> paths.shape
    (1000, 11)
    """
    rng         = np.random.default_rng(seed)
    innovations = rng.normal(loc=drift, scale=sigma, size=(n_paths, steps))
    cumulative  = np.cumsum(innovations, axis=1)
    paths       = np.column_stack([
        np.full(n_paths, start_value),
        start_value + cumulative,
    ])
    return paths
