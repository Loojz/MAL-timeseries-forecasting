# Load environment variables from .env file at project root
# Must be called before any other imports that use env vars
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="MAL · THWS Würzburg",
    page_icon="https://www.thws.de/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THWS Corporate Design ─────────────────────────────────────────────────────
# Primary brand colour: #EE750D  (source: thws.de fhws.css)
# Background family:   warm charcoal  (#1c1c1c / #242424)
# Typography:          IBM Plex Mono (headings/metrics) + IBM Plex Sans (body)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"]   { font-family: 'IBM Plex Sans', sans-serif; }
.block-container             { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3, h4              { font-family: 'IBM Plex Mono', monospace !important; }
hr                           { border-color: #3a3a3a !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebarContent"] {
    background: #181818;
    border-right: 2px solid #EE750D;
}

/* ── Metric tiles ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #242424;
    border: 1px solid #3a3a3a;
    border-left: 3px solid #EE750D;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
}
[data-testid="stMetric"] label {
    color: #9a9490 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #F0EDE8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 2px solid #EE750D !important;
    color: #EE750D !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary:hover {
    color: #EE750D !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    border: 1px solid #EE750D !important;
    color: #EE750D !important;
    background: transparent !important;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
}
[data-testid="stButton"] > button:hover {
    background: #EE750D !important;
    color: #1c1c1c !important;
}

/* ── THWS accent strip on top ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #EE750D 0%, #C5620A 100%);
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

from src.views import gold, bitcoin, eur_usd, multivariate
from src.utils.config import ZEITRAEUME

SEITEN = {
    "Teil 3: Multivariate Analyse":  multivariate.render,
    "Teil 2: Gold (ARIMA)":          gold.render,
    "Teil 2: Bitcoin (ARIMA)":       bitcoin.render,
    "Teil 2: EUR/USD (ARIMA)":       eur_usd.render,
}

st.sidebar.markdown("""
<div style='margin-bottom:0.2rem;'>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:1.05rem;
               font-weight:600;color:#EE750D;letter-spacing:0.04em;'>
    THWS
  </span>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;
               color:#9a9490;letter-spacing:0.04em;'>
    &nbsp;· MAL
  </span>
</div>
<div style='font-family:"IBM Plex Sans",sans-serif;font-size:0.7rem;
            color:#6b6560;line-height:1.6;margin-bottom:0.5rem;'>
  Zeitreihenanalyse &amp; Prognose
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("**Box-Jenkins · VAR · Granger · ETS**")
st.sidebar.markdown("---")

seite = st.sidebar.radio(
    "Navigation", list(SEITEN.keys()),
    label_visibility="collapsed",
    key="nav_radio",
)

# ── Sauberer Seitenwechsel: bei neuem Ziel einmal neu laden ──────────────────
# Verhindert, dass Streamlit alte UI-Elemente beim Seitenwechsel
# kurzzeitig zusammen mit dem neuen Inhalt anzeigt.
if st.session_state.get("_active_page") != seite:
    st.session_state["_active_page"] = seite
    st.rerun()

zeitraum_label = st.sidebar.selectbox("Zeitraum", list(ZEITRAEUME.keys()), index=2)
zeitraum       = ZEITRAEUME[zeitraum_label]

st.sidebar.markdown("---")
if st.sidebar.button("Cache leeren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small style='color:#6b6560;line-height:1.8;font-family:"IBM Plex Sans",sans-serif;'>
Assets: Gold · Bitcoin · EUR/USD<br>
Teil 2: Box-Jenkins ARIMA<br>
Teil 3: VAR · Granger · ETS<br>
Daten: Yahoo Finance · 5 Jahre
</small>
""", unsafe_allow_html=True)

SEITEN[seite](zeitraum=zeitraum, zeitraum_label=zeitraum_label)
