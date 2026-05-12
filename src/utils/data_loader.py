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


def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance DataFrame to a clean DatetimeIndex + float columns.

    yfinance >= 0.2.x returns MultiLevel column headers.  After a CSV
    round-trip those extra header rows ('Ticker', 'Date') land as ordinary
    data rows and the index becomes a plain string Index.  This helper
    handles both the live-download case and the cached-CSV case:

    Live download
        columns is a pd.MultiIndex like ('Close', 'GC=F'), ('High', 'GC=F')…
        → flatten to first level: 'Close', 'High', …
        index is already a DatetimeIndex — keep it.

    Cached CSV
        columns are plain strings (first CSV header row 'Close', 'High', …)
        index is a plain str Index containing 'Ticker', 'Date', '2015-01-02', …
        → drop non-date rows, parse the rest as DatetimeIndex.
    """
    df = df.copy()

    # 1. Flatten MultiLevel columns (live download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Normalise column names to plain strings
    df.columns = [str(c) for c in df.columns]

    # 3. Convert index to DatetimeIndex; non-parseable rows ('Ticker', 'Date')
    #    become NaT and are dropped.  format="ISO8601" avoids the pandas 2.x
    #    "Could not infer format" UserWarning for standard YYYY-MM-DD strings.
    df.index = pd.to_datetime(df.index, format="ISO8601", errors="coerce")
    df = df.loc[df.index.notna()]

    # 4. Coerce all value columns to float (spurious string rows may have
    #    forced object dtype during CSV read)
    df = df.apply(pd.to_numeric, errors="coerce")

    df.index.name = "Date"
    return df


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
        raw = pd.read_csv(file_path, index_col=0)
        return _fix_index(raw)

    print(f"[data_loader] Downloading {ticker} from Yahoo Finance ({start} to {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            f"Check the ticker symbol or your internet connection."
        )

    # Normalise before saving so the CSV always has clean single-level headers
    df = _fix_index(df)
    df.to_csv(file_path)
    print(f"[data_loader] Saved to: {file_path.name}")
    return df


if __name__ == "__main__":
    # Quick sanity check — runs only if you execute this file directly
    print("Testing data loader with BTC-USD...")
    btc = load_series("BTC-USD")
    print(f"Loaded {len(btc)} rows.")
    print(btc.head())
