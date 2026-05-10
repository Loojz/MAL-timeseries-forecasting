import pandas as pd
import requests


FRED_API_KEY = "2f1231bb70179d40a1d28f8dbc80feb4"

FRED_TICKERS = {
    "EURUSD=X": "DEXUSEU",
    "GC=F": "GOLDAMGBD228NLBM",
    "BTC-USD": None,
}


def load_series(ticker: str, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Download daily data from FRED (Federal Reserve St. Louis).

    Parameters
    ----------
    ticker : str
        Yahoo-style ticker or FRED series ID (e.g. 'EURUSD=X', 'DEXUSEU')
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and Close column
    """
    fred_id = FRED_TICKERS.get(ticker, ticker)

    if fred_id is None:
        raise ValueError(f"No FRED series available for '{ticker}'. Use yfinance instead.")

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={fred_id}"
        f"&observation_start={start}"
        f"&observation_end={end}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    records = [
        {"Date": obs["date"], "Close": float(obs["value"])}
        for obs in data["observations"]
        if obs["value"] != "."
    ]

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.dropna()

    return df