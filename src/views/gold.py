# src/views/gold.py
import streamlit as st
from src.views._univariat_base import render_univariat
from src.utils.config import TICKER, T


def render(zeitraum: str, zeitraum_label: str):
    st.title("🥇 Gold (XAU/USD) – Univariate Zeitreihenanalyse")
    st.caption("Box-Jenkins Methode · ARIMA · Strukturbrüche · Quelle: Yahoo Finance (GC=F)")
    render_univariat(
        asset_key="Gold",
        ticker=TICKER["Gold"],
        farbe=T["gold"],
        einheit="USD/oz",
        zeitraum=zeitraum,
        zeitraum_label=zeitraum_label,
    )
