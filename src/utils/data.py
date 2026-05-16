# src/utils/data.py

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from time import sleep

warnings.filterwarnings("ignore")

DATA_RAW       = os.path.join("data", "raw")
DATA_PROCESSED = os.path.join("data", "processed")


# ── Datenladen ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def lade_zeitreihe(ticker: str, period: str) -> pd.DataFrame:
    """Lädt OHLCV-Daten via yfinance. Cache: 5 Minuten."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                if "Date" not in df.columns and "Datetime" in df.columns:
                    df = df.rename(columns={"Datetime": "Date"})
                df["Date"] = pd.to_datetime(df["Date"])
                return df.dropna(subset=["Close"])
        except Exception:
            if attempt < 2:
                sleep(2 ** attempt)
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def lade_alle_zeitreihen(tickers: dict, period: str) -> dict:
    """Lädt alle Ticker als dict {name: DataFrame}."""
    result = {}
    for name, ticker in tickers.items():
        df = lade_zeitreihe(ticker, period)
        if not df.empty:
            result[name] = df
    return result


# ── Transformationen ──────────────────────────────────────────────────────────

def berechne_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Tägliche Log-Returns: r_t = ln(P_t / P_{t-1})
    Stationäre Transformation für I(1) Prozesse.
    """
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def erste_differenz(series: pd.Series) -> pd.Series:
    """Erste Differenz: Δy_t = y_t - y_{t-1}"""
    return series.diff().dropna()


def normiere(df: pd.DataFrame) -> pd.Series:
    """Normiert Close auf Index 100 für Vergleichsplot."""
    s = df["Close"].dropna()
    return s / s.iloc[0] * 100


def train_test_split_ts(series: pd.Series, train_ratio: float = 0.70):
    """
    Zeitreihen-konformer Train/Test-Split (kein zufälliges Shuffling!).
    Trainingsdaten: erste train_ratio% der Beobachtungen.
    Testdaten: letzte (1-train_ratio)% der Beobachtungen.
    """
    split = int(len(series) * train_ratio)
    return series.iloc[:split], series.iloc[split:]


# ── Deskriptive Statistik ─────────────────────────────────────────────────────

def deskriptive_statistik(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet deskriptive Statistik für Close-Preise und Log-Returns.
    Gibt DataFrame mit Kennzahlen zurück.
    """
    close   = df["Close"].dropna()
    returns = berechne_log_returns(df)
    rollmax = close.cummax()
    max_dd  = float(((close - rollmax) / rollmax).min()) * 100
    sharpe  = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0

    return pd.DataFrame({
        "Kennzahl": [
            "Mittelwert (Close)", "Median (Close)", "Standardabweichung",
            "Minimum", "Maximum", "Schiefe (Skewness)", "Wölbung (Kurtosis)",
            "Ø Log-Return (täglich)", "Std Log-Return",
            "Vola annualisiert (%)", "Sharpe Ratio (ann., rf=0)",
            "Max Drawdown (%)", "Anzahl Beobachtungen",
        ],
        "Wert": [
            round(float(close.mean()), 4),
            round(float(close.median()), 4),
            round(float(close.std()), 4),
            round(float(close.min()), 4),
            round(float(close.max()), 4),
            round(float(returns.skew()), 4),
            round(float(returns.kurtosis()), 4),
            f"{returns.mean()*100:.4f}%",
            f"{returns.std()*100:.4f}%",
            f"{returns.std()*np.sqrt(252)*100:.2f}%",
            round(sharpe, 4),
            f"{max_dd:.2f}%",
            len(close),
        ],
    })


# ── Stationaritätstests ───────────────────────────────────────────────────────

def adf_test(series: pd.Series, bezeichnung: str = "") -> dict:
    """
    Augmented Dickey-Fuller Test.
    H0: Einheitswurzel (nicht stationär) vs. H1: Stationär
    Lag-Auswahl via AIC (automatisch).
    """
    from statsmodels.tsa.stattools import adfuller
    series = series.dropna()
    if len(series) < 20:
        return {"fehler": "Zu wenige Datenpunkte (min. 20 nötig)"}
    try:
        res        = adfuller(series, autolag="AIC")
        stat, p    = res[0], res[1]
        krit       = res[4]
        stationaer = p < 0.05
        return {
            "Bezeichnung":           bezeichnung,
            "ADF Teststatistik":     round(stat, 4),
            "p-Wert":                round(p, 4),
            "Krit. Wert 1%":         round(krit["1%"], 4),
            "Krit. Wert 5%":         round(krit["5%"], 4),
            "Krit. Wert 10%":        round(krit["10%"], 4),
            "Stationär (p < 0.05)":  "Ja" if stationaer else "Nein",
            "Interpretation":        (
                "H0 abgelehnt → I(0): stationär"
                if stationaer else
                "H0 nicht abgelehnt → I(1) oder höher: differenzieren"
            ),
        }
    except Exception as e:
        return {"fehler": str(e)}


def kpss_test(series: pd.Series, bezeichnung: str = "") -> dict:
    """
    KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin).
    H0: Stationär vs. H1: Einheitswurzel
    Komplementär zum ADF-Test – beide zusammen verwenden.
    """
    from statsmodels.tsa.stattools import kpss
    series = series.dropna()
    try:
        stat, p, lags, krit = kpss(series, regression="c", nlags="auto")
        stationaer = p > 0.05
        return {
            "Bezeichnung":           bezeichnung,
            "KPSS Teststatistik":    round(stat, 4),
            "p-Wert":                round(p, 4),
            "Krit. Wert 1%":         round(krit["1%"], 4),
            "Krit. Wert 5%":         round(krit["5%"], 4),
            "Krit. Wert 10%":        round(krit["10%"], 4),
            "Stationär (p > 0.05)":  "Ja" if stationaer else "Nein",
            "Interpretation":        (
                "H0 nicht abgelehnt → stationär"
                if stationaer else
                "H0 abgelehnt → nicht stationär"
            ),
        }
    except Exception as e:
        return {"fehler": str(e)}


def ordnung_der_integration(series: pd.Series) -> int:
    """
    Bestimmt Integrationsordnung d via sequenziellem ADF-Test.
    Gibt erstes d zurück bei dem H0 (Einheitswurzel) abgelehnt wird.
    """
    from statsmodels.tsa.stattools import adfuller
    for d in range(3):
        s = series.copy()
        for _ in range(d):
            s = s.diff().dropna()
        _, p, *_ = adfuller(s.dropna(), autolag="AIC")
        if p < 0.05:
            return d
    return 2


# ── Evaluationsmetriken ───────────────────────────────────────────────────────

def berechne_metriken(y_true: pd.Series, y_pred: np.ndarray, modell_name: str = "") -> dict:
    """
    Berechnet Prognosegütemetriken (Folie 11).
    MSE, RMSE, MAE, MAPE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse    = float(np.mean((y_true - y_pred) ** 2))
    rmse   = float(np.sqrt(mse))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    # MAPE: vermeidet Division durch 0
    mask   = np.abs(y_true) > 1e-10
    mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "Modell": modell_name,
        "MSE":    round(mse, 8),
        "RMSE":   round(rmse, 8),
        "MAE":    round(mae, 8),
        "MAPE (%)": round(mape, 4),
    }


# ── Parquet ───────────────────────────────────────────────────────────────────

def speichere_parquet(df: pd.DataFrame, name: str, processed: bool = False) -> str:
    ordner = DATA_PROCESSED if processed else DATA_RAW
    os.makedirs(ordner, exist_ok=True)
    pfad = os.path.join(ordner, f"{name}.parquet")
    df.to_parquet(pfad, index=False, engine="pyarrow")
    return pfad


def lade_parquet(name: str, processed: bool = False) -> pd.DataFrame:
    ordner = DATA_PROCESSED if processed else DATA_RAW
    pfad   = os.path.join(ordner, f"{name}.parquet")
    if not os.path.exists(pfad):
        return pd.DataFrame()
    df = pd.read_parquet(pfad, engine="pyarrow")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df
