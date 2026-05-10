"""
Shared preprocessing and statistical test utilities for the MAL Time Series project.

All functions here are designed to be reusable across team members' notebooks
(Gold, BTC, EUR/USD). Import from src.utils.preprocessing.
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


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


def kpss_test(series: pd.Series, title: str = "") -> dict:
    """
    Run the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on a time series.

    H0: The series is stationary (trend-stationary).
    H1: The series has a unit root (non-stationary).

    We FAIL to reject H0 (i.e. conclude stationary) when p-value > 0.05.
    This is the opposite direction to ADF, so both tests together give
    stronger confirmation.

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
              'Critical Values', 'Stationary'

    Examples
    --------
    >>> from src.utils.preprocessing import kpss_test
    >>> result = kpss_test(gold_diff, title="Gold (1. Differenz)")
    """
    # regression="c" tests for stationarity around a constant (level stationarity)
    result = kpss(series.dropna(), regression="c", nlags="auto")

    output = {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Critical Values": result[3],
        # KPSS: stationary when we fail to reject H0, i.e. p-value > 0.05
        "Stationary": result[1] > 0.05,
    }

    header = f"KPSS-Test: {title}" if title else "KPSS-Test"
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


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot ACF and PACF side by side for a given time series.

    Useful for identifying the MA order (q) from the ACF and the AR order (p)
    from the PACF. The shaded blue band marks the 95% confidence interval —
    lags inside the band are not statistically significant.

    Parameters
    ----------
    series : pd.Series
        The (stationary) time series to analyse. NaN values are dropped.
    lags : int, optional
        Number of lags to display. Default is 40.
    title : str or None, optional
        Overall figure title shown above both subplots.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from src.utils.preprocessing import plot_acf_pacf
    >>> plot_acf_pacf(gold_diff, lags=40, title="ACF und PACF — Gold (1. Differenz)")
    """
    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series.dropna(), lags=lags, ax=ax_acf, zero=False)
    ax_acf.set_title("ACF (Autokorrelationsfunktion)", fontsize=12)
    ax_acf.set_xlabel("Lag", fontsize=10)
    ax_acf.set_ylabel("Korrelation", fontsize=10)
    ax_acf.grid(True, linestyle="--", alpha=0.4)

    # method="ywm" (Yule-Walker with bias correction) avoids spurious
    # negative values at lag 1 that appear with the default OLS estimator
    plot_pacf(series.dropna(), lags=lags, ax=ax_pacf, zero=False, method="ywm")
    ax_pacf.set_title("PACF (Partielle Autokorrelationsfunktion)", fontsize=12)
    ax_pacf.set_xlabel("Lag", fontsize=10)
    ax_pacf.set_ylabel("Korrelation", fontsize=10)
    ax_pacf.grid(True, linestyle="--", alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()
    return fig
