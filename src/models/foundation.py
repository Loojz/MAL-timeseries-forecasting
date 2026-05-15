"""
Foundation Models for Time Series Forecasting.

Chronos (Amazon) — zero-shot, local, no API key required.
TimeGPT (Nixtla)  — zero-shot, API-based (NIXTLA_API_KEY required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronos_forecast(
    series: pd.Series,
    steps: int = 10,
    model_size: str = "tiny",
    n_samples: int = 20,
    seed: int = 42,
) -> dict:
    """
    Zero-shot forecast using Amazon Chronos (local, no API key).

    Parameters
    ----------
    series : pd.Series
        Historical time series (prices or log-returns).
    steps : int
        Number of steps to forecast forward.
    model_size : str
        One of: tiny, mini, small, base, large.
        tiny  = fastest, least accurate (~80MB)
        small = good balance (~300MB)
    n_samples : int
        Number of sample paths for uncertainty quantification.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: median, lower_80, upper_80, lower_95, upper_95,
              modell, schritte, raw_samples
        On error: {"fehler": "<message>"}
    """
    try:
        import torch
        from chronos import ChronosPipeline

        torch.manual_seed(seed)

        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{model_size}",
            device_map="cpu",
            # Chronos >= 2.x uses `dtype` instead of `torch_dtype`
            dtype=torch.float32,
        )

        context = torch.tensor(
            series.dropna().values[-512:],  # Chronos context window max
            dtype=torch.float32,
        ).unsqueeze(0)

        # In Chronos >= 2.x predict() returns Tensor(batch, n_samples, steps)
        forecast = pipeline.predict(
            context,
            prediction_length=steps,
            num_samples=n_samples,
        )

        samples = forecast[0].numpy()  # shape: (n_samples, steps)

        return {
            "median":      pd.Series(np.median(samples, axis=0)),
            "lower_80":    pd.Series(np.percentile(samples, 10, axis=0)),
            "upper_80":    pd.Series(np.percentile(samples, 90, axis=0)),
            "lower_95":    pd.Series(np.percentile(samples,  5, axis=0)),
            "upper_95":    pd.Series(np.percentile(samples, 95, axis=0)),
            "modell":      f"Chronos-{model_size}",
            "schritte":    steps,
            "raw_samples": samples,
        }

    except ImportError:
        return {
            "fehler": (
                "chronos-forecasting nicht installiert.\n"
                "Installieren: pip install chronos-forecasting"
            )
        }
    except Exception as e:
        return {"fehler": f"Chronos Fehler: {str(e)}"}


def timegpt_forecast(
    series: pd.Series,
    steps: int = 10,
    api_key: str | None = None,
    freq: str = "B",
    model: str = "timegpt-2.1",
) -> dict:
    """
    Zero-shot forecast using TimeGPT-2.1 via Nixtla API.
    Requires NIXTLA_API_KEY environment variable or api_key parameter.
    Uses the TimeGPT-2 family base URL (api-preview.nixtla.io).

    Parameters
    ----------
    series : pd.Series
        Historical time series with DatetimeIndex.
    steps : int
        Number of steps to forecast.
    api_key : str, optional
        Nixtla API key. Falls back to NIXTLA_API_KEY env var.
    freq : str
        Pandas frequency string. "B" = business days.
    model : str
        TimeGPT model to use. Options:
        - "timegpt-2.1"     (latest, recommended)
        - "timegpt-2-pro"   (highest accuracy)
        - "timegpt-2-mini"  (fastest)
        - "timegpt-2-lab"   (experimental)

    Returns
    -------
    dict
        Keys: median, lower_80, upper_80, lower_95, upper_95,
              forecast_df, modell, schritte
        On error: {"fehler": "<message>"}
    """
    import os
    key = api_key or os.environ.get("NIXTLA_API_KEY")
    if not key:
        return {
            "fehler": (
                "NIXTLA_API_KEY nicht gesetzt.\n"
                "Lege eine .env Datei im Projektordner an:\n"
                "NIXTLA_API_KEY=dein_key_hier"
            )
        }

    try:
        from nixtla import NixtlaClient

        # TimeGPT-2 family requires the preview base URL
        client = NixtlaClient(
            base_url="https://api-preview.nixtla.io",
            api_key=key,
        )

        # Build DataFrame in Nixtla format
        if isinstance(series.index, pd.DatetimeIndex):
            dates = series.index
        else:
            dates = pd.date_range(
                end=pd.Timestamp.today(),
                periods=len(series),
                freq=freq,
            )

        df = pd.DataFrame({
            "ds": dates,
            "y":  series.values,
        }).dropna()

        # Nixtla requires a perfectly regular frequency — no gaps allowed.
        # Financial series have holes on public holidays (e.g. Christmas)
        # that fall on weekdays.  Reindex to the full "B" range and fill
        # holiday gaps with 0.0 (no trading → no log-return).
        df = (
            df
            .drop_duplicates(subset="ds")
            .sort_values("ds")
            .set_index("ds")
            .reindex(
                pd.date_range(
                    start=df["ds"].iloc[0],
                    end=df["ds"].iloc[-1],
                    freq=freq,
                )
            )
            .fillna(0.0)
            .reset_index()
            .rename(columns={"index": "ds"})
        )

        # Run forecast with confidence intervals
        fc = client.forecast(
            df=df,
            h=steps,
            freq=freq,
            time_col="ds",
            target_col="y",
            model=model,
            level=[80, 95],
        )

        return {
            "median":      pd.Series(fc["TimeGPT"].values),
            "lower_80":    pd.Series(
                fc.get("TimeGPT-lo-80", fc["TimeGPT"]).values
            ),
            "upper_80":    pd.Series(
                fc.get("TimeGPT-hi-80", fc["TimeGPT"]).values
            ),
            "lower_95":    pd.Series(
                fc.get("TimeGPT-lo-95", fc["TimeGPT"]).values
            ),
            "upper_95":    pd.Series(
                fc.get("TimeGPT-hi-95", fc["TimeGPT"]).values
            ),
            "forecast_df": fc,
            "modell":      "TimeGPT-2.1",
            "schritte":    steps,
        }

    except ImportError:
        return {
            "fehler": (
                "nixtla nicht installiert oder Version zu alt.\n"
                "Installieren: pip install 'nixtla>=0.7.0'"
            )
        }
    except Exception as e:
        return {"fehler": f"TimeGPT Fehler: {str(e)}"}


def evaluate_foundation_model(
    result: dict,
    y_true: pd.Series,
    modell_name: str = "",
) -> dict:
    """
    Evaluate foundation model forecast against actual values.
    Uses berechne_metriken from src.utils.data.

    Parameters
    ----------
    result : dict
        Output from chronos_forecast() or timegpt_forecast().
    y_true : pd.Series
        Actual observed values (same length as forecast).
    modell_name : str
        Label for the model in the metrics output.

    Returns
    -------
    dict with MSE, RMSE, MAE (MAPE excluded — unreliable on returns)
    """
    from src.utils.data import berechne_metriken

    if "fehler" in result:
        return {"Modell": modell_name, "fehler": result["fehler"]}

    try:
        n = min(len(result["median"]), len(y_true))
        met = berechne_metriken(
            y_true.values[:n],
            result["median"].values[:n],
            modell_name or result.get("modell", "Foundation"),
        )
        # Remove MAPE — unreliable on log-returns near zero
        met.pop("MAPE (%)", None)
        met.pop("MAPE", None)
        return met
    except Exception as e:
        return {"Modell": modell_name, "fehler": str(e)}
