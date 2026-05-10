"""
ARIMA modelling utilities for the MAL Time Series Forecasting project.

All functions are reusable across team members' notebooks (Gold, BTC, EUR/USD).
Import from src.models.arima.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
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
    model = SARIMAX(
        series.dropna(),
        order=order,
        trend="n",                     # no deterministic trend term
        enforce_stationarity=False,    # allow fitting even at boundary
        enforce_invertibility=False,   # same for MA roots
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
