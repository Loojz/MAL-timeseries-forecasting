# src/views/multivariate.py
# Teil 3: Multivariate Zeitreihenanalyse
# VAR(p) Modell · Granger Causality Test · Rekursive Prognose · State Space (ETS)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.utils.data import lade_alle_zeitreihen, berechne_log_returns, normiere
from src.utils.charts import base_layout, linie
from src.utils.config import TICKER, ANZEIGE_NAMEN, ASSET_FARBEN, T, ARIMA_PERIOD, FORECAST_STEPS
from src.models.var_model import (bereite_var_daten_vor, var_lag_selektion,
                                   var_modell_fitten, granger_causality_test,
                                   var_prognose, var_evaluation, state_space_ets)


def render(zeitraum: str, zeitraum_label: str):
    st.title("📊 Multivariate Zeitreihenanalyse")
    st.caption(
        "Teil 3 – VAR(p) Modelle · Granger Causality · Rekursive Prognose · "
        "State Space Models (ETS) · Modellvergleich"
    )

    # ── Daten laden ───────────────────────────────────────────────────────────
    with st.spinner("Lade alle Zeitreihen (5 Jahre)..."):
        alle = lade_alle_zeitreihen(TICKER, ARIMA_PERIOD)

    if len(alle) < 2:
        st.error("Mindestens 2 Zeitreihen nötig.")
        return

    # ── Normierter Vergleich ──────────────────────────────────────────────────
    st.subheader("📈 Normierter Vergleich aller Assets (Start=100)")
    fig = go.Figure()
    for name, df in alle.items():
        y = normiere(df)
        fig.add_trace(linie(
            x=df["Date"].iloc[-len(y):], y=y.values,
            name=ANZEIGE_NAMEN.get(name, name),
            color=ASSET_FARBEN.get(name, T["text"]),
            width=2,
        ))
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color=T["border"])
    fig.update_layout(**base_layout(
        title="Normierter Vergleich · 5 Jahre (Start=100)",
        yaxis_title="Index", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=T["text"]))))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── VAR-Analyse starten ───────────────────────────────────────────────────
    st.subheader("⚙️ VAR-Modell Pipeline")
    st.markdown("""
    **VAR(p) Prozess** (Folie 24):

    $$\\mathbf{y}_t = \\mathbf{c} + \\Phi_1 \\mathbf{y}_{t-1} + \\cdots + \\Phi_p \\mathbf{y}_{t-p} + \\boldsymbol{\\epsilon}_t$$

    mit $\\boldsymbol{\\epsilon}_t \\sim iid(0, \\Sigma_\\epsilon)$ und $\\mathbf{y}_t = [r_{Gold,t}, r_{BTC,t}, r_{EUR,t}]'$
    """)

    if st.button("🚀 VAR-Analyse starten", key="var_start"):
        # Daten vorbereiten
        with st.spinner("Bereite Daten vor..."):
            df_returns = bereite_var_daten_vor(alle)
            st.info(
                f"Gemeinsamer Datensatz: {len(df_returns)} Beobachtungen · "
                f"Variablen: {list(df_returns.columns)}"
            )

        # ── Stationaritätstest ────────────────────────────────────────────────
        st.markdown("### 1. Stationaritätstest (VAR erfordert I(0))")
        from src.utils.data import adf_test
        adf_rows = []
        for col in df_returns.columns:
            res = adf_test(df_returns[col], f"{col} Log-Returns")
            adf_rows.append({
                "Variable":             col,
                "ADF Teststatistik":    res.get("ADF Teststatistik", "–"),
                "p-Wert":               res.get("p-Wert", "–"),
                "Stationär (p<0.05)":   res.get("Stationär (p < 0.05)", "–"),
            })
        st.dataframe(pd.DataFrame(adf_rows), use_container_width=True, hide_index=True)
        st.caption("Log-Returns sind I(0) – Voraussetzung für stationäres VAR erfüllt.")

        # ── Lag-Selektion ─────────────────────────────────────────────────────
        st.markdown("### 2. Lag-Selektion via AIC / BIC / HQIC")
        with st.spinner("Berechne Informationskriterien..."):
            lag_res = var_lag_selektion(df_returns, max_lags=10)

        st.dataframe(lag_res["tabelle"], use_container_width=True, hide_index=True)
        st.success(lag_res["interpretation"])
        opt_lag = lag_res["empfohlen"]
        st.metric("Empfohlene Lag-Länge p", opt_lag)

        # ── VAR Modell fitten ─────────────────────────────────────────────────
        st.markdown(f"### 3. VAR({opt_lag}) Modell")
        with st.spinner(f"Fitte VAR({opt_lag})..."):
            var_res = var_modell_fitten(df_returns, opt_lag)

        st.text(str(var_res.summary())[:3000])  # Summary ausgeben

        # ── Granger Causality ─────────────────────────────────────────────────
        st.markdown("### 4. Granger Causality Test")
        st.markdown("""
        **Granger (1969)**: $x$ Granger-verursacht $y$, wenn $x$ zur Vorhersage von $y$ beiträgt.

        $H_0$: $\\Phi_{k,xy} = 0 \\ \\forall k$ (keine Granger-Kausalität)
        $H_1$: Mindestens ein $\\Phi_{k,xy} \\neq 0$

        Wald-Test auf untere Dreiecksstruktur der Koeffizientenmatrix.
        """)
        with st.spinner("Berechne Granger Causality Tests..."):
            gc_res = granger_causality_test(df_returns, opt_lag)

        if not gc_res["tabelle"].empty:
            st.dataframe(gc_res["tabelle"], use_container_width=True, hide_index=True)

            # Heatmap der Granger-Kausalität
            vars_list = df_returns.columns.tolist()
            n = len(vars_list)
            gc_matrix = np.zeros((n, n))
            for i, caused in enumerate(vars_list):
                for j, causing in enumerate(vars_list):
                    if causing != caused:
                        key  = f"{causing} → {caused}"
                        pval = gc_res["details"].get(key, {}).get("p-Wert", 1.0)
                        gc_matrix[i, j] = 1 - float(pval) if isinstance(pval, (int, float)) else 0

            fig_gc = go.Figure(go.Heatmap(
                z=gc_matrix,
                x=vars_list, y=vars_list,
                colorscale=[[0, T["card_bg"]], [0.5, T["purple"]], [1, T["up"]]],
                zmin=0, zmax=1,
                text=[[f"p={gc_res['details'].get(f'{vars_list[j]} → {vars_list[i]}', {}).get('p-Wert', '–')}"
                       for j in range(n)] for i in range(n)],
                texttemplate="%{text}",
                textfont=dict(size=11, color="#fff", family=T["font"]),
                colorbar=dict(title=dict(text="1-p", font=dict(color=T["text"])),
                              tickfont=dict(color=T["text"])),
                hovertemplate="<b>%{y} ← %{x}</b><br>Stärke: %{z:.3f}<extra></extra>",
                xgap=2, ygap=2,
            ))
            fig_gc.update_layout(**base_layout(
                title="Granger Causality – Heatmap (Spalte → Zeile)",
                height=350, hovermode="closest",
                xaxis_title="Verursachend (causing)",
                yaxis_title="Verursacht (caused)"))
            st.plotly_chart(fig_gc, use_container_width=True)

        # ── VAR Evaluation ────────────────────────────────────────────────────
        st.markdown("### 5. Modell-Evaluation (Train/Test 70/30)")
        with st.spinner("Evaluiere VAR-Modell..."):
            eval_res = var_evaluation(df_returns, opt_lag)

        if "fehler" not in eval_res:
            rows = []
            for col, met in eval_res["metriken"].items():
                rows.append({
                    "Asset":      col,
                    "Modell":     "VAR",
                    "RMSE":       met["VAR"]["RMSE"],
                    "MAE":        met["VAR"]["MAE"],
                    "MAPE (%)":   met["VAR"]["MAPE (%)"],
                })
                rows.append({
                    "Asset":      col,
                    "Modell":     "Random Walk",
                    "RMSE":       met["RandomWalk"]["RMSE"],
                    "MAE":        met["RandomWalk"]["MAE"],
                    "MAPE (%)":   met["RandomWalk"]["MAPE (%)"],
                })
            df_eval = pd.DataFrame(rows)
            st.dataframe(df_eval, use_container_width=True, hide_index=True)

        # ── VAR Prognose ──────────────────────────────────────────────────────
        st.markdown(f"### 6. Rekursive VAR({opt_lag}) Prognose – {FORECAST_STEPS} Perioden")
        st.markdown(
            r"$\hat{\mathbf{y}}_{t+1} = \hat{\mathbf{c}} + \hat{\Phi}_1 \mathbf{y}_t + \cdots + \hat{\Phi}_p \mathbf{y}_{t-p+1}$"
        )
        with st.spinner("Berechne Prognosen..."):
            prog_res = var_prognose(var_res, steps=FORECAST_STEPS)

        for col in df_returns.columns:
            color = ASSET_FARBEN.get(col, T["text"])
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=list(range(max(0, len(df_returns)-60), len(df_returns))),
                y=df_returns[col].iloc[-60:].values * 100,
                name="Historisch", line=dict(color=color, width=2)))
            fc_idx = list(range(len(df_returns), len(df_returns) + FORECAST_STEPS))
            fig_p.add_trace(go.Scatter(
                x=fc_idx, y=prog_res["prognose"][col].values * 100,
                name="Prognose", line=dict(color=T["up"], width=2, dash="dash")))
            fig_p.add_trace(go.Scatter(
                x=fc_idx + fc_idx[::-1],
                y=list(prog_res["upper_95"][col].values * 100) +
                  list(prog_res["lower_95"][col].values[::-1] * 100),
                fill="toself", fillcolor="rgba(34,197,94,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="95% KI"))
            fig_p.add_hline(y=0, line_width=1, line_dash="dot", line_color=T["border"])
            fig_p.update_layout(**base_layout(
                title=f"VAR({opt_lag}) Prognose – {ANZEIGE_NAMEN.get(col, col)}",
                yaxis_title="Log-Return (%)", height=300))
            st.plotly_chart(fig_p, use_container_width=True)

        st.divider()

        # ── State Space Models ────────────────────────────────────────────────
        st.subheader("🔮 7. State Space Models (ETS)")
        st.markdown("""
        **State Space Framework** (Folie 31):

        - **Observation equation**: $y_t = A_t x_t + v_t$
        - **State equation**: $x_t = \\Phi x_{t-1} + w_t$

        **ETS** (Error-Trend-Seasonal) ist eine praktische Implementierung.
        Smoothing-Parameter $\\alpha$ wird optimal geschätzt.
        """)

        ets_metriken = []
        for asset_name, df_asset in alle.items():
            log_ret_asset = berechne_log_returns(df_asset)
            anzeige = ANZEIGE_NAMEN.get(asset_name, asset_name)
            color   = ASSET_FARBEN.get(asset_name, T["text"])

            ets_res = state_space_ets(log_ret_asset, anzeige, FORECAST_STEPS)
            if "fehler" not in ets_res:
                ets_metriken.append({
                    "Asset":     anzeige,
                    "α (ETS)":   ets_res["alpha"],
                    "RMSE":      ets_res["metriken"]["RMSE"],
                    "MAE":       ets_res["metriken"]["MAE"],
                    "MAPE (%)":  ets_res["metriken"]["MAPE (%)"],
                })

        if ets_metriken:
            st.dataframe(pd.DataFrame(ets_metriken), use_container_width=True, hide_index=True)

        # ── Gesamtübersicht Modellvergleich ───────────────────────────────────
        st.divider()
        st.subheader("🏆 Gesamtübersicht: Modellvergleich")
        st.markdown(
            "Vergleich aller Modelle nach RMSE auf dem Test-Set (30%):\n"
            "Random Walk (Benchmark) · ARIMA (univariat) · VAR (multivariat) · ETS (State Space)"
        )

        if "fehler" not in eval_res and ets_metriken:
            summary_rows = []
            for col in df_returns.columns:
                anzeige = ANZEIGE_NAMEN.get(col, col)
                rw_rmse = eval_res["metriken"][col]["RandomWalk"]["RMSE"]
                var_rmse = eval_res["metriken"][col]["VAR"]["RMSE"]
                ets_row = next((e for e in ets_metriken if e["Asset"] == anzeige), {})
                ets_rmse = ets_row.get("RMSE", "–")
                summary_rows.append({
                    "Asset":            anzeige,
                    "Random Walk RMSE": rw_rmse,
                    "VAR RMSE":         var_rmse,
                    "ETS RMSE":         ets_rmse,
                    "Bestes Modell":    min(
                        [("Random Walk", rw_rmse), ("VAR", var_rmse),
                         ("ETS", ets_rmse if isinstance(ets_rmse, float) else 999)],
                        key=lambda x: x[1]
                    )[0],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.caption(
                "RMSE = √MSE · kleinere Werte = bessere Prognose · "
                "Random Walk ist der naive Benchmark (H₀: Einheitswurzel)"
            )
