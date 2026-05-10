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
    "bg":          "#0b0d17",
    "paper_bg":    "#0b0d17",
    "card_bg":     "#111827",
    "grid":        "#1f2937",
    "border":      "#374151",
    "text":        "#9ca3af",
    "text_bright": "#e5e7eb",
    "font":        "IBM Plex Mono, monospace",
    "gold":        "#f59e0b",
    "btc":         "#f97316",
    "eur_usd":     "#3b82f6",
    "up":          "#22c55e",
    "dn":          "#ef4444",
    "purple":      "#8b5cf6",
}

ASSET_FARBEN = {
    "Gold":    T["gold"],
    "Bitcoin": T["btc"],
    "EUR_USD": T["eur_usd"],
}
