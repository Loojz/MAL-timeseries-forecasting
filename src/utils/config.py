# src/utils/config.py

TICKER = {
    "Gold":    "GC=F",
    "Bitcoin": "BTC-USD",
    "EUR_USD": "EURUSD=X",
}

ANZEIGE_NAMEN = {
    "Gold":    "Gold (XAU/USD)",
    "Bitcoin": "Bitcoin (BTC/USD)",
    "EUR_USD": "EUR/USD",
}

ZEITRAEUME = {
    "1 Jahr":   "1y",
    "2 Jahre":  "2y",
    "5 Jahre":  "5y",
    "10 Jahre": "10y",
}

# Fixe Parameter für Analyse
ARIMA_PERIOD   = "5y"   # Datenbasis für alle Modelle
TRAIN_RATIO    = 0.70   # 70% Train, 30% Test
FORECAST_STEPS = 10     # h-step ahead forecast

T = {
    # ── THWS dark background family (warm charcoal, not cold blue-black) ─────
    "bg":          "#1c1c1c",   # main background
    "paper_bg":    "#1c1c1c",
    "card_bg":     "#242424",   # card / metric tile background
    "grid":        "#2d2d2d",   # chart grid lines
    "border":      "#3a3a3a",   # dividers and subtle borders
    # ── Typography ────────────────────────────────────────────────────────────
    "text":        "#9a9490",   # muted label text (warm grey)
    "text_bright": "#F0EDE8",   # heading / value text (warm white)
    "font":        "IBM Plex Mono, monospace",
    # ── THWS brand orange (source: thws.de fhws.css) ─────────────────────────
    "accent":      "#EE750D",   # primary THWS orange — use for highlights,
    "accent_dark": "#C5620A",   # borders, active states, focus rings
    # ── Asset colours ────────────────────────────────────────────────────────
    "gold":        "#F5A623",   # gold/amber — distinguishable from THWS orange
    "btc":         "#F97316",   # bitcoin warm orange
    "eur_usd":     "#4A9EE0",   # EUR/USD blue
    # ── Status / chart ───────────────────────────────────────────────────────
    "up":          "#22c55e",   # positive return
    "dn":          "#ef4444",   # negative return
    "purple":      "#8b5cf6",   # TimeGPT / secondary model colour
}

ASSET_FARBEN = {
    "Gold":    T["gold"],
    "Bitcoin": T["btc"],
    "EUR_USD": T["eur_usd"],
}
