import yfinance as yf
import pandas as pd


def load_series(ticker: str, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'EURUSD=X', 'GC=F', 'BTC-USD')
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and OHLCV columns
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    return df