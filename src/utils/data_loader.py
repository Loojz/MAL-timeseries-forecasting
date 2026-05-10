"""
Data loader for the MAL Time Series Forecasting project.

Provides a single function to load financial time series from Yahoo Finance.
Data is frozen to a fixed date range and cached locally as CSV files,
so that all team members work on identical data and forecasts remain
reproducible across runs.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Frozen date range — DO NOT change without team agreement
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2025-05-04"

# Path to the raw data directory (resolved relative to project root)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_series(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load a financial time series for a given ticker.

    On first call: downloads from Yahoo Finance and caches as CSV.
    On subsequent calls: reads from the cached CSV (fast, reproducible).

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "BTC-USD", "EURUSD=X", "CL=F".
    start : str, optional
        Start date in YYYY-MM-DD format. Defaults to project-wide START_DATE.
    end : str, optional
        End date in YYYY-MM-DD format. Defaults to project-wide END_DATE.
    force_refresh : bool, optional
        If True, re-downloads from Yahoo Finance even if a cached file exists.

    Returns
    -------
    pd.DataFrame
        Time series with a DatetimeIndex and columns Open, High, Low,
        Close, Adj Close, Volume.

    Examples
    --------
    >>> btc = load_series("BTC-USD")
    >>> oil = load_series("CL=F")
    >>> eurusd = load_series("EURUSD=X")
    """
    # Build a deterministic filename so the same call always hits the same file
    safe_ticker = ticker.replace("=", "_").replace("-", "_").replace("/", "_")
    filename = f"{safe_ticker}_{start}_to_{end}.csv"
    file_path = RAW_DATA_DIR / filename

    # Make sure the data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not force_refresh:
        print(f"[data_loader] Loading cached file: {file_path.name}")
        # yfinance >=0.2 writes two metadata rows (Ticker, Date) before the data.
        # skiprows=[1, 2] drops them so we get a clean DatetimeIndex + float columns.
        return pd.read_csv(file_path, skiprows=[1, 2], index_col=0, parse_dates=True)

    print(f"[data_loader] Downloading {ticker} from Yahoo Finance ({start} to {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            f"Check the ticker symbol or your internet connection."
        )

    df.to_csv(file_path)
    print(f"[data_loader] Saved to: {file_path.name}")
    return df


if __name__ == "__main__":
    # Quick sanity check — runs only if you execute this file directly
    print("Testing data loader with BTC-USD...")
    btc = load_series("BTC-USD")
    print(f"Loaded {len(btc)} rows.")
    print(btc.head())
