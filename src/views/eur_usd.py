# src/views/eur_usd.py
import streamlit as st
from src.views._univariat_base import render_univariat
from src.utils.config import TICKER, T


def render(zeitraum: str, zeitraum_label: str):
    st.title("💱 EUR/USD – Univariate Zeitreihenanalyse")
    st.caption("Box-Jenkins Methode · ARIMA · Strukturbrüche · Quelle: Yahoo Finance (EURUSD=X)")
    render_univariat(
        asset_key="EUR_USD",
        ticker=TICKER["EUR_USD"],
        farbe=T["eur_usd"],
        einheit="USD pro EUR",
        zeitraum=zeitraum,
        zeitraum_label=zeitraum_label,
    )
