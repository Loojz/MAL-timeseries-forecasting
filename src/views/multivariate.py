# src/views/multivariate.py
# Teil 3: Multivariate Zeitreihenanalyse
# VAR(p) Modell · Granger Causality Test · Rekursive Prognose · State Space (ETS)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.utils.data import (lade_alle_zeitreihen, berechne_log_returns,
                             normiere, berechne_metriken)
from src.utils.charts import base_layout, linie
from src.utils.config import TICKER, ANZEIGE_NAMEN, ASSET_FARBEN, T, FORECAST_STEPS
from src.models.var_model import (bereite_var_daten_vor, var_lag_selektion,
                                   var_modell_fitten, granger_causality_test,
                                   var_prognose, var_evaluation, state_space_ets)


def render(zeitraum: str, zeitraum_label: str):
    st.title("Multivariate Zeitreihenanalyse")
    st.caption(
        "Teil 3 – VAR(p) Modelle · Granger Causality · Rekursive Prognose · "
        "State Space Models (ETS) · Modellvergleich"
    )

    # ── Daten laden ───────────────────────────────────────────────────────────
    with st.spinner("Lade alle Zeitreihen ..."):
        alle = lade_alle_zeitreihen(TICKER, zeitraum)

    if len(alle) < 2:
        st.error("Mindestens 2 Zeitreihen nötig.")
        return

    tab_main, tab_research = st.tabs(["Analyse", "Research"], key="mv_tabs")

    # ═══════════════════════════════════════════════════════════════════════════
    with tab_main:
    # ═══════════════════════════════════════════════════════════════════════════

        # ── Normierter Vergleich ──────────────────────────────────────────────
        st.subheader("Normierter Vergleich aller Assets — Start = 100")
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
            title=f"Normierter Vergleich · {zeitraum_label} (Start=100)",
            yaxis_title="Index", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                        font=dict(color=T["text"]))))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── VAR-Analyse starten ───────────────────────────────────────────────
        st.subheader("⚙️ VAR-Modell Pipeline")
        st.markdown("""
        **VAR(p) Prozess** (Folie 24):

        $$\\mathbf{y}_t = \\mathbf{c} + \\Phi_1 \\mathbf{y}_{t-1} + \\cdots + \\Phi_p \\mathbf{y}_{t-p} + \\boldsymbol{\\epsilon}_t$$

        mit $\\boldsymbol{\\epsilon}_t \\sim iid(0, \\Sigma_\\epsilon)$ und $\\mathbf{y}_t = [r_{Gold,t}, r_{BTC,t}, r_{EUR,t}]'$
        """)

        if st.button("VAR-Analyse starten", key="var_start"):
            # Daten vorbereiten
            with st.spinner("Bereite Daten vor..."):
                df_returns = bereite_var_daten_vor(alle)
                st.info(
                    f"Gemeinsamer Datensatz: {len(df_returns)} Beobachtungen · "
                    f"Variablen: {list(df_returns.columns)}"
                )

            # ── Stationaritätstest ────────────────────────────────────────────
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

            # ── Lag-Selektion ─────────────────────────────────────────────────
            st.markdown("### 2. Lag-Selektion via AIC / BIC / HQIC")
            with st.spinner("Berechne Informationskriterien..."):
                lag_res = var_lag_selektion(df_returns, max_lags=10)

            st.dataframe(lag_res["tabelle"], use_container_width=True, hide_index=True)
            st.success(lag_res["interpretation"])
            opt_lag = lag_res["empfohlen"]
            st.metric("Empfohlene Lag-Länge p", opt_lag)

            # ── VAR Modell fitten ─────────────────────────────────────────────
            st.markdown(f"### 3. VAR({opt_lag}) Modell")
            with st.spinner(f"Fitte VAR({opt_lag})..."):
                var_res = var_modell_fitten(df_returns, opt_lag)

            st.text(str(var_res.summary())[:3000])  # Summary ausgeben

            # ── Granger Causality ─────────────────────────────────────────────
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

            # ── VAR Evaluation ────────────────────────────────────────────────
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

            # ── VAR Prognose ──────────────────────────────────────────────────
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

            # ── State Space Models ────────────────────────────────────────────
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

            # ── ARIMA (univariat, pro Asset) ──────────────────────────────────
            st.divider()
            st.subheader("8. ARIMA — Univariate Modelle")
            st.markdown(
                "Box-Jenkins ARIMA wird für jedes Asset separat auf "
                "Log-Renditen geschätzt. Automatische Modellwahl via AIC/BIC."
            )

            arima_metriken  = []

            for asset_name, df_asset in alle.items():
                anzeige    = ANZEIGE_NAMEN.get(asset_name, asset_name)
                color      = ASSET_FARBEN.get(asset_name, T["text"])
                log_ret    = berechne_log_returns(df_asset).dropna()
                # box_jenkins_pipeline expects raw close prices — it computes
                # log-returns internally. Passing pre-computed log-returns would
                # trigger a second log-transform, producing garbage for Gold/EUR-USD.
                close_series = df_asset["Close"].dropna()

                with st.spinner(f"Fitte ARIMA für {anzeige}..."):
                    try:
                        from src.models.arima_model import box_jenkins_pipeline
                        arima_res = box_jenkins_pipeline(
                            close_series, anzeige,
                            forecast_steps=FORECAST_STEPS,
                        )
                    except Exception as e:
                        st.warning(f"ARIMA ({anzeige}) Fehler: {e}")
                        arima_metriken.append({
                            "Asset":         anzeige,
                            "Bestes Modell": "–",
                            "RMSE":          "–",
                            "MAE":           "–",
                        })
                        continue

                # Extract values using correct keys
                best_order = (
                    arima_res.get("bestes_order")
                    or arima_res.get("schritt4_selektion", {}).get("bestes_order", (0, 1, 0))
                )

                prognose  = arima_res.get("schritt7_prognose", {})
                fc_values = prognose.get("prognose_mean",   None)
                fc_lower  = prognose.get("konfidenz_lower", None)
                fc_upper  = prognose.get("konfidenz_upper", None)

                benchmark = arima_res.get("benchmark_vergleich", {})
                arima_met = benchmark.get("arima", {})
                rmse_val  = arima_met.get("RMSE", "–")
                mae_val   = arima_met.get("MAE",  "–")

                arima_metriken.append({
                    "Asset":         anzeige,
                    "Bestes Modell": f"ARIMA{best_order}",
                    "RMSE":          rmse_val,
                    "MAE":           mae_val,
                })

                # Forecast plot
                if fc_values is not None and len(fc_values) > 0:
                    hist_vals = log_ret.iloc[-60:].values * 100
                    hist_x    = list(range(len(hist_vals)))
                    fc_len    = len(fc_values)
                    fc_x      = list(range(
                        len(hist_vals),
                        len(hist_vals) + fc_len,
                    ))

                    fig_a = go.Figure()

                    # History
                    fig_a.add_trace(go.Scatter(
                        x=hist_x, y=hist_vals,
                        name="Historisch",
                        line=dict(color=color, width=2),
                    ))

                    # ARIMA forecast
                    fig_a.add_trace(go.Scatter(
                        x=fc_x,
                        y=fc_values.values * 100,
                        name=f"ARIMA{best_order}",
                        line=dict(color=T["purple"], width=2, dash="dash"),
                        mode="lines+markers",
                        marker=dict(size=4),
                    ))

                    # 95% CI
                    if fc_lower is not None and fc_upper is not None:
                        fig_a.add_trace(go.Scatter(
                            x=fc_x + fc_x[::-1],
                            y=list(fc_upper.values * 100)
                              + list(fc_lower.values[::-1] * 100),
                            fill="toself",
                            fillcolor="rgba(139,92,246,0.12)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="95% KI",
                        ))

                    fig_a.add_hline(
                        y=0, line_width=1, line_dash="dot",
                        line_color=T["border"],
                    )
                    fig_a.add_vline(
                        x=len(hist_vals) - 0.5,
                        line_width=1, line_dash="dash",
                        line_color="rgba(255,255,255,0.3)",
                        annotation_text="Forecast Start",
                        annotation_position="top left",
                    )
                    fig_a.update_layout(**base_layout(
                        title=(
                            f"ARIMA{best_order} — {anzeige} · "
                            f"{FORECAST_STEPS}-Tage Prognose"
                        ),
                        yaxis_title="Log-Return (%)",
                        height=300,
                    ))
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.info(f"ARIMA ({anzeige}): Kein Forecast verfügbar.")

            # Metrics table
            if arima_metriken:
                st.dataframe(
                    pd.DataFrame(arima_metriken),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Foundation Models (Chronos + TimeGPT) ─────────────────────────
            st.divider()
            st.subheader("9. Foundation Models — Zero-Shot Forecasting")
            st.markdown(
                "Chronos (Amazon, lokal) und TimeGPT-2.1 (Nixtla, API) werden "
                "ohne Training auf diesen Daten eingesetzt — reines Zero-Shot Forecasting."
            )

            from src.models.foundation import (
                chronos_forecast, timegpt_forecast, evaluate_foundation_model
            )
            import os

            HORIZON = FORECAST_STEPS
            foundation_metriken = []
            nixtla_key = os.environ.get("NIXTLA_API_KEY")

            for asset_name, df_asset in alle.items():
                anzeige  = ANZEIGE_NAMEN.get(asset_name, asset_name)
                color    = ASSET_FARBEN.get(asset_name, T["text"])
                log_ret  = berechne_log_returns(df_asset).dropna()
                split    = int(len(log_ret) * 0.70)
                train    = log_ret.iloc[:split]
                test     = log_ret.iloc[split:]
                y_eval   = test.iloc[:HORIZON]

                col_c, col_t = st.columns(2)

                # ── Chronos ──
                with col_c:
                    st.markdown(f"**Chronos — {anzeige}**")
                    with st.spinner(f"Chronos berechnet {anzeige}..."):
                        c_res = chronos_forecast(
                            train, steps=HORIZON,
                            model_size="tiny", n_samples=20, seed=42,
                        )

                    if "fehler" not in c_res:
                        met_c = evaluate_foundation_model(
                            c_res, y_eval, f"Chronos ({anzeige})"
                        )
                        foundation_metriken.append({
                            "Asset":  anzeige,
                            "Modell": "Chronos",
                            "RMSE":   met_c.get("RMSE", "–"),
                            "MAE":    met_c.get("MAE",  "–"),
                        })

                        hist_vals = log_ret.iloc[-60:].values * 100
                        hist_x    = list(range(len(hist_vals)))
                        fc_x      = list(range(len(hist_vals),
                                              len(hist_vals) + HORIZON))

                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(
                            x=hist_x, y=hist_vals,
                            name="Historisch",
                            line=dict(color=color, width=2),
                        ))
                        fig_c.add_trace(go.Scatter(
                            x=fc_x,
                            y=c_res["median"].values * 100,
                            name=f"Chronos ({HORIZON}d)",
                            line=dict(color="#22c55e", width=2, dash="dash"),
                            mode="lines+markers", marker=dict(size=4),
                        ))
                        fig_c.add_trace(go.Scatter(
                            x=fc_x + fc_x[::-1],
                            y=list(c_res["upper_95"].values * 100)
                              + list(c_res["lower_95"].values[::-1] * 100),
                            fill="toself",
                            fillcolor="rgba(34,197,94,0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="95% KI",
                        ))
                        fig_c.add_hline(
                            y=0, line_width=1, line_dash="dot",
                            line_color=T["border"],
                        )
                        fig_c.add_vline(
                            x=len(hist_vals) - 0.5,
                            line_width=1, line_dash="dash",
                            line_color="rgba(255,255,255,0.3)",
                        )
                        fig_c.update_layout(**base_layout(
                            title=f"Chronos — {anzeige}",
                            yaxis_title="Log-Return (%)", height=280,
                        ))
                        st.plotly_chart(fig_c, use_container_width=True)
                    else:
                        st.warning(f"Chronos: {c_res['fehler']}")

                # ── TimeGPT ──
                with col_t:
                    st.markdown(f"**TimeGPT-2.1 — {anzeige}**")
                    if not nixtla_key:
                        st.info("NIXTLA_API_KEY nicht gesetzt — TimeGPT nicht verfügbar.")
                    else:
                        with st.spinner(f"TimeGPT berechnet {anzeige}..."):
                            t_res = timegpt_forecast(
                                train, steps=HORIZON,
                                api_key=nixtla_key, model="timegpt-2.1",
                            )

                        if "fehler" not in t_res:
                            met_t = evaluate_foundation_model(
                                t_res, y_eval, f"TimeGPT ({anzeige})"
                            )
                            foundation_metriken.append({
                                "Asset":  anzeige,
                                "Modell": "TimeGPT-2.1",
                                "RMSE":   met_t.get("RMSE", "–"),
                                "MAE":    met_t.get("MAE",  "–"),
                            })

                            hist_vals = log_ret.iloc[-60:].values * 100
                            hist_x    = list(range(len(hist_vals)))
                            fc_x      = list(range(len(hist_vals),
                                                  len(hist_vals) + HORIZON))

                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(
                                x=hist_x, y=hist_vals,
                                name="Historisch",
                                line=dict(color=color, width=2),
                            ))
                            fig_t.add_trace(go.Scatter(
                                x=fc_x,
                                y=t_res["median"].values * 100,
                                name=f"TimeGPT-2.1 ({HORIZON}d)",
                                line=dict(color=T["purple"], width=2,
                                          dash="dash"),
                                mode="lines+markers", marker=dict(size=4),
                            ))
                            fig_t.add_trace(go.Scatter(
                                x=fc_x + fc_x[::-1],
                                y=list(t_res["upper_95"].values * 100)
                                  + list(t_res["lower_95"].values[::-1] * 100),
                                fill="toself",
                                fillcolor="rgba(139,92,246,0.10)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="95% KI",
                            ))
                            fig_t.add_hline(
                                y=0, line_width=1, line_dash="dot",
                                line_color=T["border"],
                            )
                            fig_t.add_vline(
                                x=len(hist_vals) - 0.5,
                                line_width=1, line_dash="dash",
                                line_color="rgba(255,255,255,0.3)",
                            )
                            fig_t.update_layout(**base_layout(
                                title=f"TimeGPT-2.1 — {anzeige}",
                                yaxis_title="Log-Return (%)", height=280,
                            ))
                            st.plotly_chart(fig_t, use_container_width=True)
                        else:
                            st.warning(f"TimeGPT: {t_res['fehler']}")

            if foundation_metriken:
                st.dataframe(
                    pd.DataFrame(foundation_metriken),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "MAPE ausgeschlossen — bei Log-Renditen nahe Null "
                    "führt Division durch ~0 zu verzerrten Werten."
                )

            # ── Gesamtübersicht Modellvergleich ───────────────────────────────
            st.divider()
            st.subheader("Gesamtübersicht — Modellvergleich")
            st.markdown(
                "RMSE aller getesteten Modelle auf dem Test-Set (30%). "
                "Random Walk ist der naive Benchmark."
            )

            if "fehler" not in eval_res:
                summary_rows = []
                for col in df_returns.columns:
                    anzeige = ANZEIGE_NAMEN.get(col, col)

                    # VAR + Random Walk from eval_res
                    rw_rmse  = eval_res["metriken"][col]["RandomWalk"]["RMSE"]
                    var_rmse = eval_res["metriken"][col]["VAR"]["RMSE"]

                    # ETS
                    ets_row  = next(
                        (e for e in ets_metriken if e["Asset"] == anzeige), {}
                    )
                    ets_rmse = ets_row.get("RMSE", "–")

                    # ARIMA
                    arima_row = next(
                        (a for a in arima_metriken if a["Asset"] == anzeige), {}
                    )
                    arima_rmse = arima_row.get("RMSE", "–")

                    # Chronos
                    chronos_row = next(
                        (f for f in foundation_metriken
                         if f["Asset"] == anzeige and f["Modell"] == "Chronos"),
                        {}
                    )
                    chronos_rmse = chronos_row.get("RMSE", "–")

                    # TimeGPT
                    tgpt_row = next(
                        (f for f in foundation_metriken
                         if f["Asset"] == anzeige
                         and f["Modell"] == "TimeGPT-2.1"),
                        {}
                    )
                    tgpt_rmse = tgpt_row.get("RMSE", "–")

                    # Best model
                    candidates = [
                        ("Random Walk", rw_rmse),
                        ("VAR",         var_rmse),
                        ("ETS",         ets_rmse),
                        ("ARIMA",       arima_rmse),
                        ("Chronos",     chronos_rmse),
                        ("TimeGPT-2.1", tgpt_rmse),
                    ]
                    valid = [
                        (name, v) for name, v in candidates
                        if isinstance(v, (int, float))
                    ]
                    best = min(valid, key=lambda x: x[1])[0] if valid else "–"

                    summary_rows.append({
                        "Asset":         anzeige,
                        "Random Walk":   rw_rmse,
                        "VAR":           var_rmse,
                        "ETS":           ets_rmse,
                        "ARIMA":         arima_rmse,
                        "Chronos":       chronos_rmse,
                        "TimeGPT-2.1":   tgpt_rmse,
                        "Bestes Modell": best,
                    })

                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "RMSE = √MSE — kleinere Werte bedeuten bessere Prognose. "
                    "Random Walk ist der naive Benchmark (H₀: Einheitswurzel). "
                    "MAPE ausgeschlossen (instabil bei Log-Renditen nahe 0)."
                )

    # ═══════════════════════════════════════════════════════════════════════════
    with tab_research:
    # ═══════════════════════════════════════════════════════════════════════════

        st.info(
            "**Research —** Öffne die Expander unten um die "
            "jeweilige Analyse zu laden. Jede Sektion berechnet sich "
            "unabhängig und nur bei Bedarf."
        )
        st.markdown("""
        ### Statistische Tiefenanalyse

        Diese Sektion enthält die vollständige statistische Analyse
        aus dem Research-Notebook (`03_multivariate_var.ipynb`).
        Die Ergebnisse sind akademisch dokumentiert und dienen als
        methodische Grundlage für die Hauptanalyse.
        """)

        # ── Gemeinsame Basis-Berechnungen (VAR(3) einmal fitten, in Exp. 3–6 wiederverwenden)
        _df   = None
        _var3 = None
        try:
            _df = bereite_var_daten_vor(alle)
            from statsmodels.tsa.vector_ar.var_model import VAR as _VAR
            _var3 = _VAR(_df).fit(3)
        except Exception:
            pass  # expanders handle _df/_var3 being None individually

        # ─────────────────────────────────────────────────────────────────────
        # Expander 1 — Johansen Kointegrations-Test
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("01 — Kointegrations-Test (Johansen)", expanded=False):
            try:
                st.markdown(
                    "Der Johansen-Test prüft, ob zwischen den drei Log-Preis-Reihen "
                    "eine langfristige Gleichgewichtsbeziehung (Kointegration) besteht. "
                    "Ist Kointegration vorhanden, wäre ein VECM anstelle von VAR geeignet. "
                    "Wir testen auf Basis der log-transformierten Preisniveaus."
                )
                with st.spinner(f"Berechne Johansen-Test ({zeitraum_label} Preisdaten)..."):
                    from statsmodels.tsa.vector_ar.vecm import coint_johansen

                    # Load price data for selected period (cached after first call)
                    alle_raw = lade_alle_zeitreihen(TICKER, zeitraum)
                    log_prices = pd.DataFrame({
                        name: np.log(df["Close"])
                        for name, df in alle_raw.items()
                    }).dropna()

                    jh = coint_johansen(log_prices, det_order=0, k_ar_diff=1)

                # Trace test results table
                h0_labels = ["r ≤ 0", "r ≤ 1", "r ≤ 2"]
                n_r = min(len(jh.lr1), len(h0_labels))
                trace_rows = []
                for i in range(n_r):
                    sig = bool(jh.lr1[i] > jh.cvt[i, 1])  # 95% critical value
                    trace_rows.append({
                        "H₀":                  h0_labels[i],
                        "Trace-Statistik":     round(float(jh.lr1[i]), 2),
                        "Krit. 90%":           round(float(jh.cvt[i, 0]), 2),
                        "Krit. 95%":           round(float(jh.cvt[i, 1]), 2),
                        "Signifikant (95%)":   "Ja" if sig else "Nein",
                    })
                st.dataframe(
                    pd.DataFrame(trace_rows),
                    use_container_width=True, hide_index=True,
                )

                n_coint = sum(jh.lr1[i] > jh.cvt[i, 1] for i in range(n_r))
                st.metric("Kointegrationsbeziehungen (95%)", n_coint)

                if n_coint == 0:
                    st.info(
                        "Kein Hinweis auf Kointegration → "
                        "VAR auf Log-Renditen ist methodisch korrekt."
                    )
                else:
                    st.warning(
                        f"{n_coint} Kointegrationsbeziehung(en) gefunden → "
                        "VECM wäre methodisch korrekt; VAR bleibt gültige Approximation."
                    )
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 2 — QLR-Test
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("02 — QLR-Test · Unbekannte Strukturbrüche", expanded=False):
            try:
                st.markdown(
                    "Der QLR-Test sucht nach einem unbekannten Strukturbruch. "
                    "Für jede der drei Reihen wird die F-Statistik über alle "
                    "möglichen Bruchpunkte (mittlere 70%) berechnet."
                )
                with st.spinner("Berechne QLR-Tests (3 Assets)..."):
                    from src.models.arima_model import qlr_test
                    import plotly.subplots as _sp

                    qlr_rows       = []
                    qlr_per_asset  = {}
                    for name, df_a in alle.items():
                        log_ret = np.log(df_a["Close"]).diff().dropna()
                        r = qlr_test(log_ret)
                        qlr_per_asset[name] = r
                        qlr_rows.append({
                            "Asset":            ANZEIGE_NAMEN.get(name, name),
                            "Max F-Statistik":  round(float(r.get("Max. F-Statistik", 0)), 4),
                            "Krit. Wert (10%)": float(r.get("Krit. Wert 10%", 7.17)),
                            "Strukturbruch":    r.get("Strukturbruch", "–"),
                        })

                st.dataframe(
                    pd.DataFrame(qlr_rows),
                    use_container_width=True, hide_index=True,
                )

                # Plotly subplots — one panel per asset
                asset_names_list = list(alle.keys())
                fig_qlr = _sp.make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[ANZEIGE_NAMEN.get(n, n) for n in asset_names_list],
                    shared_xaxes=False,
                    vertical_spacing=0.10,
                )
                for idx, name in enumerate(asset_names_list):
                    r        = qlr_per_asset[name]
                    f_stats  = r.get("f_stats", [])
                    bps      = r.get("breakpoints", list(range(len(f_stats))))
                    krit_val = float(r.get("Krit. Wert 10%", 7.17))
                    color    = ASSET_FARBEN.get(name, T["text"])
                    row_num  = idx + 1
                    if f_stats:
                        fig_qlr.add_trace(
                            go.Scatter(
                                x=bps, y=f_stats, mode="lines",
                                line=dict(color=color, width=1.5),
                                name=ANZEIGE_NAMEN.get(name, name),
                                showlegend=True,
                            ),
                            row=row_num, col=1,
                        )
                        # Dashed horizontal critical-value line
                        fig_qlr.add_shape(
                            type="line",
                            x0=bps[0], x1=bps[-1],
                            y0=krit_val, y1=krit_val,
                            line=dict(color=T["dn"], width=1.5, dash="dash"),
                            row=row_num, col=1,
                        )

                fig_qlr.update_layout(
                    height=600,
                    paper_bgcolor=T["paper_bg"],
                    plot_bgcolor=T["card_bg"],
                    font=dict(color=T["text"], family=T["font"]),
                    title_text="QLR F-Statistik — kritischer Wert 7.17 (gestrichelt)",
                    showlegend=True,
                )
                st.plotly_chart(fig_qlr, use_container_width=True)
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 3 — VAR(3) vs VAR(auto)
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("03 — VAR(3) vs. VAR(auto) · Lag-Vergleich", expanded=False):
            try:
                st.markdown(
                    "Der Prof empfiehlt VAR(3) als Standard. Wir vergleichen "
                    "das automatisch gewählte Modell (BIC) mit dem festen VAR(3)."
                )
                with st.spinner("Vergleiche VAR-Modelle..."):
                    if _df is None or _var3 is None:
                        raise ValueError("Log-Returns DataFrame nicht verfügbar.")

                    from statsmodels.tsa.vector_ar.var_model import VAR as _VAR2
                    lag_sel  = _VAR2(_df).select_order(maxlags=10)
                    p_auto   = int(lag_sel.bic)
                    var_auto = _VAR2(_df).fit(p_auto)

                cmp_rows = [
                    {"Kriterium": "AIC",
                     "VAR(auto)": round(float(var_auto.aic), 4),
                     "VAR(3)":    round(float(_var3.aic),    4)},
                    {"Kriterium": "BIC",
                     "VAR(auto)": round(float(var_auto.bic), 4),
                     "VAR(3)":    round(float(_var3.bic),    4)},
                    {"Kriterium": "HQIC",
                     "VAR(auto)": round(float(var_auto.hqic), 4),
                     "VAR(3)":    round(float(_var3.hqic),    4)},
                    {"Kriterium": "Log-Likelihood",
                     "VAR(auto)": round(float(var_auto.llf), 4),
                     "VAR(3)":    round(float(_var3.llf),    4)},
                    {"Kriterium": "Lag p",
                     "VAR(auto)": p_auto,
                     "VAR(3)":    3},
                ]
                st.dataframe(
                    pd.DataFrame(cmp_rows),
                    use_container_width=True, hide_index=True,
                )
                st.metric("Automatisch gewählter Lag (BIC)", p_auto)

                if p_auto == 3:
                    verdict = "BIC wählt p=3 — identisch mit der Professorenempfehlung"
                elif float(var_auto.bic) < float(_var3.bic):
                    verdict = (
                        f"VAR({p_auto}) hat niedrigeres BIC "
                        f"({round(float(var_auto.bic), 2)} vs. "
                        f"{round(float(_var3.bic), 2)}) — "
                        "automatische Wahl ist parsimonischer."
                    )
                else:
                    verdict = (
                        f"VAR(3) hat niedrigeres BIC "
                        f"({round(float(_var3.bic), 2)} vs. "
                        f"{round(float(var_auto.bic), 2)}) — "
                        "Professorenempfehlung ist optimal."
                    )
                st.info(verdict)
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 4 — Impulse Response Functions (IRF)
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("04 — Impulse Response Functions (IRF)", expanded=False):
            try:
                st.markdown(
                    "Die IRF zeigt wie ein einmaliger Schock in Asset X die anderen "
                    "Assets über die nächsten 20 Perioden beeinflusst. Wir verwenden "
                    "orthogonalisierte IRFs via Cholesky-Zerlegung. "
                    "Reihenfolge: [Gold, BTC, EUR/USD]."
                )
                with st.spinner("Berechne IRF (VAR(3), 20 Perioden)..."):
                    if _var3 is None:
                        raise ValueError("VAR(3) nicht verfügbar.")

                    import matplotlib.pyplot as plt
                    from io import BytesIO

                    irf     = _var3.irf(periods=20)
                    fig_irf = irf.plot(orth=True, figsize=(12, 10))
                    if fig_irf is None:
                        fig_irf = plt.gcf()
                    plt.suptitle("Impulse Response Functions — VAR(3)", y=1.02)
                    plt.tight_layout()
                    buf = BytesIO()
                    fig_irf.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    plt.close(fig_irf)

                st.image(buf, use_container_width=True)

                # Economic interpretation table
                interp_rows = [
                    {
                        "Schock in":  "Gold",
                        "Effekt auf": "EUR/USD",
                        "Ökonomische Interpretation":
                            "Goldpreisanstieg → USD-Schwäche (Gold als USD-Hedge)",
                    },
                    {
                        "Schock in":  "BTC",
                        "Effekt auf": "EUR/USD",
                        "Ökonomische Interpretation":
                            "BTC als Risk-On/Risk-Off Indikator für USD",
                    },
                    {
                        "Schock in":  "EUR/USD",
                        "Effekt auf": "Gold",
                        "Ökonomische Interpretation":
                            "Währungsbewegung → Goldnachfrage (Dollar-Transmissionskanal)",
                    },
                ]
                st.dataframe(
                    pd.DataFrame(interp_rows),
                    use_container_width=True, hide_index=True,
                )
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 5 — FEVD
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("05 — Forecast Error Variance Decomposition (FEVD)", expanded=False):
            try:
                st.markdown(
                    "Die FEVD zeigt, wie viel der Vorhersageunsicherheit eines Assets "
                    "durch Schocks aus anderen Assets erklärt wird (bei h=10 Perioden)."
                )
                with st.spinner("Berechne FEVD (VAR(3), h=10)..."):
                    if _var3 is None or _df is None:
                        raise ValueError("VAR(3) nicht verfügbar.")

                    fevd = _var3.fevd(periods=10)
                    # Shape: (n_assets, periods, n_assets)
                    # decomp[:, -1, :] = last period = h=10 (index 9)
                    fevd_df = pd.DataFrame(
                        fevd.decomp[:, -1, :],
                        index=list(_df.columns),
                        columns=list(_df.columns),
                    ).round(4) * 100  # convert to percent

                # Rename rows/columns to display names
                fevd_display = fevd_df.copy()
                fevd_display.index   = [ANZEIGE_NAMEN.get(c, c) for c in fevd_df.index]
                fevd_display.columns = [ANZEIGE_NAMEN.get(c, c) for c in fevd_df.columns]

                st.dataframe(
                    fevd_display.style.format("{:.2f}%"),
                    use_container_width=True,
                )

                # Dynamically find the strongest cross-asset spillover
                max_spill = 0.0
                max_shock, max_target = "", ""
                for row_asset in fevd_df.index:
                    for col_asset in fevd_df.columns:
                        if row_asset != col_asset:
                            val = float(fevd_df.loc[row_asset, col_asset])
                            if val > max_spill:
                                max_spill  = val
                                max_shock  = ANZEIGE_NAMEN.get(col_asset, col_asset)
                                max_target = ANZEIGE_NAMEN.get(row_asset, row_asset)

                st.info(
                    f"▸ {max_target} wird zu ~{max_spill:.1f}% durch {max_shock} erklärt — "
                    "der stärkste Cross-Asset Spillover im System."
                )
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 6 — Vollständiges Backtesting (4 Modelle)
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("06 — Vollständiges Backtesting · 4 Modelle", expanded=False):
            try:
                st.markdown(
                    "Vergleich von VAR(3), VAR(auto), ARIMA und Random Walk auf "
                    "einem 30%-Holdout-Set."
                )
                with st.spinner("Berechne Backtesting (4 Modelle × 3 Assets)..."):
                    if _df is None or _var3 is None:
                        raise ValueError("Log-Returns DataFrame nicht verfügbar.")

                    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                    from statsmodels.tsa.vector_ar.var_model import VAR as _VAR3
                    from src.models.arima_model import box_jenkins_pipeline

                    split = int(len(_df) * 0.70)
                    train = _df.iloc[:split]
                    test  = _df.iloc[split:]

                    # --- VAR(3) ---
                    var3_bt    = _VAR3(train).fit(3)
                    var3_fc    = var3_bt.forecast(train.values[-3:], steps=len(test))
                    var3_fc_df = pd.DataFrame(var3_fc, index=test.index, columns=_df.columns)

                    # --- VAR(auto) via BIC ---
                    lag_sel_bt = _VAR3(train).select_order(maxlags=10)
                    p_auto_bt  = max(1, int(lag_sel_bt.bic))
                    varo_bt    = _VAR3(train).fit(p_auto_bt)
                    varo_fc    = varo_bt.forecast(train.values[-p_auto_bt:], steps=len(test))
                    varo_fc_df = pd.DataFrame(varo_fc, index=test.index, columns=_df.columns)

                    # --- ARIMA per asset: use box_jenkins_pipeline to get best order ---
                    arima_orders = {}
                    for name in _df.columns:
                        try:
                            bj = box_jenkins_pipeline(
                                alle[name]["Close"], name,
                                forecast_steps=1, max_p=2, max_q=2,
                            )
                            arima_orders[name] = bj["schritt4_selektion"]["bestes_order"]
                        except Exception:
                            arima_orders[name] = (0, 0, 1)  # fallback: MA(1)

                    arima_fc = {}
                    for name in _df.columns:
                        order = arima_orders[name]
                        try:
                            mod = _ARIMA(train[name], order=order).fit()
                            arima_fc[name] = mod.forecast(steps=len(test)).values
                        except Exception:
                            arima_fc[name] = np.full(len(test), train[name].mean())

                    # --- Random Walk: last training value per asset ---
                    rw_fc = {
                        name: np.full(len(test), train[name].iloc[-1])
                        for name in _df.columns
                    }

                # Build master comparison table — MSE, RMSE, MAE (no MAPE)
                all_rows = []
                for name in _df.columns:
                    y_true      = test[name].values
                    label       = ANZEIGE_NAMEN.get(name, name)
                    arima_label = f"ARIMA{arima_orders.get(name, (0, 0, 1))}"
                    for model_label, y_pred in [
                        ("VAR(3)",              var3_fc_df[name].values),
                        (f"VAR({p_auto_bt})",   varo_fc_df[name].values),
                        (arima_label,           arima_fc[name]),
                        ("Random Walk",         rw_fc[name]),
                    ]:
                        m = berechne_metriken(y_true, y_pred, model_label)
                        all_rows.append({
                            "Asset":  label,
                            "Modell": m["Modell"],
                            "MSE":    m["MSE"],
                            "RMSE":   m["RMSE"],
                            "MAE":    m["MAE"],
                        })

                master = (
                    pd.DataFrame(all_rows)
                    .sort_values(["Asset", "RMSE"])
                    .reset_index(drop=True)
                )
                st.dataframe(master, use_container_width=True, hide_index=True)
                st.caption(
                    "MAPE wurde ausgeschlossen — bei Log-Renditen nahe Null "
                    "führt Division durch sehr kleine Werte zu verzerrten Werten."
                )

                # Winner summary — 3 metric columns
                asset_labels = master["Asset"].unique().tolist()
                cols_bt = st.columns(min(3, len(asset_labels)))
                for i, asset_label in enumerate(asset_labels[:3]):
                    sub      = master[master["Asset"] == asset_label]
                    best_row = sub.loc[sub["RMSE"].idxmin()]
                    with cols_bt[i]:
                        st.metric(
                            f"{asset_label} — Bestes Modell",
                            best_row["Modell"],
                            delta=f"RMSE: {float(best_row['RMSE']):.5f}",
                        )
            except Exception as e:
                st.warning(f"Berechnung nicht möglich: {str(e)}")

        # ─────────────────────────────────────────────────────────────────────
        # Expander 7 — Foundation Models (Chronos + TimeGPT)
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Foundation Models — Zero-Shot Forecasting")
        st.markdown("""
Foundation Models sind große vortrainierte neuronale Netze, die ohne
Feintuning auf neuen Zeitreihen Prognosen erstellen können
(**Zero-Shot Learning**). Wir testen zwei Modelle:

| Modell | Anbieter | Typ | Besonderheit |
|--------|----------|-----|--------------|
| **Chronos** | Amazon | Lokal, kostenlos | T5-Transformer, auf Millionen Zeitreihen trainiert |
| **TimeGPT** | Nixtla | API-basiert | Speziell für Zeitreihen entwickelt |
        """)

        with st.expander("07 — Foundation Models · Chronos + TimeGPT-2.1", expanded=False):
            try:
                import os
                from src.models.foundation import (
                    chronos_forecast, timegpt_forecast, evaluate_foundation_model,
                )

                if _df is None:
                    raise ValueError(
                        "Log-Returns DataFrame nicht verfügbar — "
                        "bitte Seite neu laden."
                    )

                # ── Model size selector ───────────────────────────────────────
                model_size = st.selectbox(
                    "Chronos Modellgröße",
                    options=["tiny", "mini", "small"],
                    index=0,
                    help="tiny = schnellste (~80 MB), small = genauer (~300 MB)",
                )

                # Short horizon — Chronos is optimised for 10-50 steps, not 780
                CHRONOS_HORIZON = 10

                foundation_rows = []

                # ── Chronos — one plot per asset ──────────────────────────────
                st.markdown("#### Chronos (Amazon) — Lokal")
                st.caption(
                    f"Chronos wird beim ersten Aufruf heruntergeladen "
                    f"(~80 MB für '{model_size}'). Bitte warten."
                )

                for asset_name, color in ASSET_FARBEN.items():
                    anzeige      = ANZEIGE_NAMEN.get(asset_name, asset_name)
                    asset_logret = _df[asset_name].dropna()

                    # 70/30 split — evaluate only on first CHRONOS_HORIZON test days
                    split     = int(len(asset_logret) * 0.70)
                    train_ret = asset_logret.iloc[:split]
                    test_ret  = asset_logret.iloc[split:]
                    y_eval    = test_ret.iloc[:CHRONOS_HORIZON]

                    st.markdown(f"**{anzeige}**")

                    with st.spinner(f"Chronos berechnet {anzeige}..."):
                        chronos_res = chronos_forecast(
                            train_ret,
                            steps=CHRONOS_HORIZON,
                            model_size=model_size,
                            n_samples=20,
                            seed=42,
                        )

                    if "fehler" in chronos_res:
                        st.warning(f"Chronos ({anzeige}): {chronos_res['fehler']}")
                        continue

                    # Evaluate against the first CHRONOS_HORIZON test days
                    met = evaluate_foundation_model(
                        chronos_res, y_eval,
                        f"Chronos-{model_size} ({anzeige})",
                    )
                    foundation_rows.append({
                        "Asset":  anzeige,
                        "Modell": f"Chronos-{model_size}",
                        "MSE":    met.get("MSE", "–"),
                        "RMSE":   met.get("RMSE", "–"),
                        "MAE":    met.get("MAE", "–"),
                    })

                    # Plot: last 60 days of history + CHRONOS_HORIZON forecast steps
                    hist_vals = asset_logret.iloc[-60:].values * 100
                    hist_x    = list(range(len(hist_vals)))
                    fc_x      = list(range(len(hist_vals),
                                           len(hist_vals) + CHRONOS_HORIZON))

                    fig_c = go.Figure()

                    # Historical log-returns
                    fig_c.add_trace(go.Scatter(
                        x=hist_x, y=hist_vals,
                        name="Historisch (letzte 60 Tage)",
                        line=dict(color=color, width=2),
                    ))

                    # Chronos median with markers
                    fig_c.add_trace(go.Scatter(
                        x=fc_x,
                        y=chronos_res["median"].values * 100,
                        name=f"Chronos Median ({CHRONOS_HORIZON} Tage)",
                        line=dict(color=T["up"], width=2, dash="dash"),
                        mode="lines+markers",
                        marker=dict(size=5),
                    ))

                    # 80% CI (inner band)
                    fig_c.add_trace(go.Scatter(
                        x=fc_x + fc_x[::-1],
                        y=list(chronos_res["upper_80"].values * 100)
                          + list(chronos_res["lower_80"].values[::-1] * 100),
                        fill="toself",
                        fillcolor="rgba(34,197,94,0.20)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="80%-KI",
                    ))

                    # 95% CI (outer band)
                    fig_c.add_trace(go.Scatter(
                        x=fc_x + fc_x[::-1],
                        y=list(chronos_res["upper_95"].values * 100)
                          + list(chronos_res["lower_95"].values[::-1] * 100),
                        fill="toself",
                        fillcolor="rgba(34,197,94,0.10)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="95%-KI",
                    ))

                    # Zero line and forecast boundary marker
                    fig_c.add_hline(
                        y=0, line_width=1, line_dash="dot",
                        line_color="rgba(255,255,255,0.3)",
                    )
                    fig_c.add_vline(
                        x=len(hist_vals) - 0.5,
                        line_width=1, line_dash="dash",
                        line_color="rgba(255,255,255,0.4)",
                        annotation_text="Forecast Start",
                        annotation_position="top left",
                    )
                    fig_c.update_layout(**base_layout(
                        title=f"Chronos — {anzeige} | {CHRONOS_HORIZON}-Tage Zero-Shot Forecast",
                        yaxis_title="Log-Return (%)",
                        height=320,
                    ))
                    st.plotly_chart(fig_c, use_container_width=True)
                    st.caption(
                        f"Chronos forecasted {CHRONOS_HORIZON} Handelstage vorwärts "
                        f"(Zero-Shot, kein Training auf diesen Daten). "
                        f"Evaluation gegen die ersten {CHRONOS_HORIZON} Tage des Test-Sets."
                    )

                # ── TimeGPT-2.1 ───────────────────────────────────────────────
                st.divider()
                st.markdown("#### TimeGPT-2.1 (Nixtla) — API")

                nixtla_key = os.environ.get("NIXTLA_API_KEY")
                if not nixtla_key:
                    st.info(
                        "**TimeGPT API-Key nicht gefunden.**\n\n"
                        "Lege eine `.env` Datei im Projektordner an:\n\n"
                        "```\nNIXTLA_API_KEY=dein_key_hier\n```\n\n"
                        "Dann die App neu starten. "
                        "API-Key erhalten: [dashboard.nixtla.io](https://dashboard.nixtla.io)"
                    )
                else:
                    timegpt_model = st.selectbox(
                        "TimeGPT Modell",
                        options=["timegpt-2.1", "timegpt-2-pro", "timegpt-2-mini"],
                        index=0,
                        help=(
                            "timegpt-2.1 = empfohlen | "
                            "timegpt-2-pro = höchste Genauigkeit | "
                            "timegpt-2-mini = schnellste"
                        ),
                    )

                    for asset_name, color in ASSET_FARBEN.items():
                        anzeige      = ANZEIGE_NAMEN.get(asset_name, asset_name)
                        asset_logret = _df[asset_name].dropna()
                        split        = int(len(asset_logret) * 0.70)
                        train_ret    = asset_logret.iloc[:split]
                        test_ret     = asset_logret.iloc[split:]
                        y_eval       = test_ret.iloc[:CHRONOS_HORIZON]

                        st.markdown(f"**{anzeige}**")

                        with st.spinner(f"TimeGPT-2.1 berechnet {anzeige}..."):
                            tgpt_res = timegpt_forecast(
                                train_ret,
                                steps=CHRONOS_HORIZON,
                                api_key=nixtla_key,
                                model=timegpt_model,
                            )

                        if "fehler" in tgpt_res:
                            st.warning(f"TimeGPT ({anzeige}): {tgpt_res['fehler']}")
                            continue

                        # Evaluate against first CHRONOS_HORIZON test days
                        met_t = evaluate_foundation_model(
                            tgpt_res, y_eval,
                            f"TimeGPT-2.1 ({anzeige})"
                        )
                        foundation_rows.append({
                            "Asset":  anzeige,
                            "Modell": "TimeGPT-2.1",
                            "MSE":    met_t.get("MSE",  "–"),
                            "RMSE":   met_t.get("RMSE", "–"),
                            "MAE":    met_t.get("MAE",  "–"),
                        })

                        # Plot — same style as Chronos
                        full_series = asset_logret
                        hist_vals   = full_series.iloc[-60:].values * 100
                        hist_x      = list(range(len(hist_vals)))
                        fc_x        = list(range(
                            len(hist_vals),
                            len(hist_vals) + CHRONOS_HORIZON
                        ))

                        fig_t = go.Figure()

                        # History
                        fig_t.add_trace(go.Scatter(
                            x=hist_x, y=hist_vals,
                            name="Historisch (letzte 60 Tage)",
                            line=dict(color=color, width=2),
                        ))

                        # TimeGPT median
                        fig_t.add_trace(go.Scatter(
                            x=fc_x,
                            y=tgpt_res["median"].values * 100,
                            name=f"TimeGPT-2.1 Median ({CHRONOS_HORIZON} Tage)",
                            line=dict(color=T["purple"], width=2, dash="dash"),
                            mode="lines+markers",
                            marker=dict(size=5),
                        ))

                        # 80% CI
                        fig_t.add_trace(go.Scatter(
                            x=fc_x + fc_x[::-1],
                            y=list(tgpt_res["upper_80"].values * 100)
                              + list(tgpt_res["lower_80"].values[::-1] * 100),
                            fill="toself",
                            fillcolor="rgba(139,92,246,0.20)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="80%-KI",
                        ))

                        # 95% CI
                        fig_t.add_trace(go.Scatter(
                            x=fc_x + fc_x[::-1],
                            y=list(tgpt_res["upper_95"].values * 100)
                              + list(tgpt_res["lower_95"].values[::-1] * 100),
                            fill="toself",
                            fillcolor="rgba(139,92,246,0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="95%-KI",
                        ))

                        fig_t.add_hline(
                            y=0, line_width=1, line_dash="dot",
                            line_color="rgba(255,255,255,0.3)",
                        )
                        fig_t.add_vline(
                            x=len(hist_vals) - 0.5,
                            line_width=1, line_dash="dash",
                            line_color="rgba(255,255,255,0.4)",
                            annotation_text="Forecast Start",
                            annotation_position="top left",
                        )
                        fig_t.update_layout(
                            **base_layout(
                                title=(
                                    f"TimeGPT-2.1 — {anzeige} | "
                                    f"{CHRONOS_HORIZON}-Tage Zero-Shot Forecast"
                                ),
                                yaxis_title="Log-Return (%)",
                                height=320,
                            )
                        )
                        st.plotly_chart(fig_t, use_container_width=True)
                        st.caption(
                            f"TimeGPT-2.1 forecasted {CHRONOS_HORIZON} Handelstage "
                            f"vorwärts (Zero-Shot). "
                            f"Evaluation gegen die ersten {CHRONOS_HORIZON} Tage "
                            f"des Test-Sets."
                        )

                # ── Summary table ─────────────────────────────────────────────
                if foundation_rows:
                    st.divider()
                    st.markdown("**Foundation Models — Evaluationsübersicht:**")
                    st.dataframe(
                        pd.DataFrame(foundation_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "MAPE ausgeschlossen — bei Log-Renditen nahe Null "
                        "führt Division durch ~0 zu verzerrten Werten."
                    )

            except Exception as e:
                st.warning(f"Foundation Models konnten nicht geladen werden: {e}")
