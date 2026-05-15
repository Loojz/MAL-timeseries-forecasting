import streamlit as st

st.set_page_config(
    page_title="MAL Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetric"] {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 0.6rem 0.9rem;
}
[data-testid="stMetric"] label {
    color: #6b7280 !important; font-size: 0.7rem !important;
    letter-spacing: 0.07em; text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #f3f4f6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebarContent"] { background: #0b0d17; border-right: 1px solid #1f2937; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
hr { border-color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)

from src.views import gold, bitcoin, eur_usd, multivariate
from src.utils.config import ZEITRAEUME

SEITEN = {
    "📊  Teil 3: Multivariate Analyse":    multivariate.render,
    "🥇  Teil 2: Gold (ARIMA)":            gold.render,
    "₿   Teil 2: Bitcoin (ARIMA)":         bitcoin.render,
    "💱  Teil 2: EUR/USD (ARIMA)":         eur_usd.render,
}

st.sidebar.markdown("## 📈 MAL Time Series")
st.sidebar.markdown("**Box-Jenkins · VAR · Granger · ETS**")
st.sidebar.markdown("---")

seite = st.sidebar.radio("Navigation", list(SEITEN.keys()), label_visibility="collapsed")

zeitraum_label = st.sidebar.selectbox("⏱ Zeitraum", list(ZEITRAEUME.keys()), index=2)
zeitraum       = ZEITRAEUME[zeitraum_label]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Cache leeren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small style='color:#4b5563;line-height:1.8;'>
Assets: Gold · Bitcoin · EUR/USD<br>
Teil 2: Box-Jenkins ARIMA<br>
Teil 3: VAR · Granger · ETS<br>
Daten: Yahoo Finance · 5 Jahre
</small>
""", unsafe_allow_html=True)

SEITEN[seite](zeitraum=zeitraum, zeitraum_label=zeitraum_label)
