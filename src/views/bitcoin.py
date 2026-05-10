# src/views/bitcoin.py
import streamlit as st
from src.views._univariat_base import render_univariat
from src.utils.config import TICKER, T


def render(zeitraum: str, zeitraum_label: str):
    st.title("₿ Bitcoin (BTC/USD) – Univariate Zeitreihenanalyse")
    st.caption("Box-Jenkins Methode · ARIMA · Strukturbrüche · Quelle: Yahoo Finance (BTC-USD)")
    render_univariat(
        asset_key="Bitcoin",
        ticker=TICKER["Bitcoin"],
        farbe=T["btc"],
        einheit="USD",
        zeitraum=zeitraum,
        zeitraum_label=zeitraum_label,
    )
