"""
Shared preprocessing and statistical test utilities for the MAL Time Series project.

All functions here are designed to be reusable across team members' notebooks
(Gold, BTC, EUR/USD). Import from src.utils.preprocessing.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, title: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test on a time series.

    H0: The series has a unit root (non-stationary).
    H1: The series has no unit root (stationary).

    Rejects H0 (i.e. concludes stationary) when p-value < 0.05.

    Parameters
    ----------
    series : pd.Series
        The time series to test. NaN values are dropped automatically.
    title : str, optional
        Label printed in the output header for readability.

    Returns
    -------
    dict
        Keys: 'Test Statistic', 'p-value', 'Lags Used',
              'Observations', 'Critical Values', 'Stationary'

    Examples
    --------
    >>> from src.utils.preprocessing import adf_test
    >>> result = adf_test(gold["Close"], title="Gold Close")
    """
    result = adfuller(series.dropna(), autolag="AIC")

    labels = ["Test Statistic", "p-value", "Lags Used", "Observations"]
    output = dict(zip(labels, result[:4]))
    output["Critical Values"] = result[4]
    output["Stationary"] = result[1] < 0.05

    header = f"ADF-Test: {title}" if title else "ADF-Test"
    print(f"\n{'=' * 50}")
    print(header)
    print(f"{'=' * 50}")
    for k, v in output.items():
        if k == "Critical Values":
            for ck, cv in v.items():
                print(f"  Kritischer Wert ({ck}):  {cv:.4f}")
        elif k == "Stationary":
            pass  # printed separately as verdict below
        else:
            print(f"  {k}: {v}")

    verdict = "STATIONAER ✓" if output["Stationary"] else "NICHT STATIONAER ✗"
    print(f"\n  Ergebnis: {verdict}  (p = {output['p-value']:.4f})")
    print(f"{'=' * 50}\n")

    return output
