# src/views/_univariat_base.py
# Gemeinsame Basis für alle univariaten ARIMA-Views.
# Ergänzt um Gold-Notebook Methodik:
# - Chow-Test auf differenzierter Reihe (korrekt)
# - Kandidatenmodelle explizit + Random Walk Benchmark
# - ADF/KPSS Diskrepanz-Erklärung
# - Beide Modelle (Haupt + Benchmark) in Residualdiagnose

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from src.utils.data import (lade_zeitreihe, deskriptive_statistik,
                              adf_test, kpss_test, berechne_log_returns)
from src.utils.charts import base_layout, linie
from src.utils.config import T, FORECAST_STEPS, TRAIN_RATIO
from src.models.arima_model import (box_jenkins_pipeline, qlr_test,
                                     chow_test_auf_diff, KANDIDATEN)


def render_univariat(asset_key, ticker, farbe, einheit, zeitraum, zeitraum_label):

    with st.spinner(f"Lade Daten..."):
        df = lade_zeitreihe(ticker, zeitraum)

    if df.empty:
        st.error("Keine Daten verfügbar.")
        return

    close   = df["Close"].dropna()
    log_ret = berechne_log_returns(df)
    diff1   = close.diff().dropna()

    # ── Metriken ──────────────────────────────────────────────────────────────
    aktuell = float(close.iloc[-1])
    perf    = (aktuell - float(close.iloc[0])) / float(close.iloc[0]) * 100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Aktueller Preis",         f"{aktuell:,.4f} {einheit}")
    c2.metric(f"Performance ({zeitraum_label})", f"{perf:+.2f}%")
    c3.metric(f"Hoch ({zeitraum_label})",        f"{float(df['High'].max()):,.4f}")
    c4.metric(f"Tief ({zeitraum_label})",        f"{float(df['Low'].min()):,.4f}")

    fig = go.Figure()
    fig.add_trace(linie(df["Date"], close, asset_key, farbe, fill=True))
    fig.update_layout(**base_layout(
        title=f"{asset_key} – Preisverlauf ({zeitraum_label})",
        yaxis_title=einheit, height=340))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 1. Deskriptive Statistik ──────────────────────────────────────────────
    st.subheader("1 — Deskriptive Statistik")
    st.dataframe(deskriptive_statistik(df), use_container_width=True, hide_index=True)

    st.divider()

    # ── 2. Strukturbrüche ─────────────────────────────────────────────────────
    st.subheader("2 — Strukturbrüche")
    st.markdown(
        "Strukturbrüche werden **vor** dem Stationaritätstest geprüft, da sie den "
        "ADF-Test verzerren können (Perron 1989)."
    )

    tab_chow, tab_qlr = st.tabs(["Chow-Test (T* bekannt)", "QLR-Test (T* unbekannt)"])

    with tab_chow:
        st.markdown(
            "**Methodik (wie im Gold-Notebook):** Chow-Test auf der **differenzierten Reihe** "
            "(Log-Returns), nicht auf dem Preisniveau. Testet ob sich der *Prozess* geändert hat.\n\n"
            "$H_0: \\beta_2 = 0$ (kein Bruch) vs. $H_1: \\beta_2 \\neq 0$"
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("**COVID-19 Schock (15.03.2020)**")
            try:
                res_covid = chow_test_auf_diff(log_ret, "2020-03-15")
                st.dataframe(
                    pd.DataFrame([{k: v for k, v in res_covid.items()
                                   if k != "Interpretation"}]).T
                    .rename(columns={0: "Ergebnis"}),
                    use_container_width=True)
                st.info(res_covid.get("Interpretation", ""))
            except Exception as e:
                st.warning(str(e))

        with col_b2:
            st.markdown("**Ukraine-Invasion (24.02.2022)**")
            try:
                res_ukr = chow_test_auf_diff(log_ret, "2022-02-24")
                st.dataframe(
                    pd.DataFrame([{k: v for k, v in res_ukr.items()
                                   if k != "Interpretation"}]).T
                    .rename(columns={0: "Ergebnis"}),
                    use_container_width=True)
                st.info(res_ukr.get("Interpretation", ""))
            except Exception as e:
                st.warning(str(e))

        # Preisverlauf mit Bruchpunkten
        fig_bp = go.Figure()
        fig_bp.add_trace(linie(df["Date"], close, asset_key, farbe))
        for datum, label, color in [("2020-03-15", "COVID-19", T["dn"]),
                                     ("2022-02-24", "Ukraine", T["eur_usd"])]:
            try:
                xval = pd.Timestamp(datum)
                fig_bp.add_vline(x=xval, line_dash="dash", line_color=color, line_width=1.5,
                                  annotation_text=label, annotation_font_color=color)
            except Exception:
                pass
        fig_bp.update_layout(**base_layout(
            title=f"{asset_key} – potenzielle Strukturbrüche", height=280))
        st.plotly_chart(fig_bp, use_container_width=True)

    with tab_qlr:
        st.markdown(
            "**QLR-Test**: Sucht automatisch den Bruchpunkt mit maximaler F-Statistik.\n"
            "Kritischer Wert 10%: 7.17 (Andrews 1993)"
        )
        if st.button("QLR-Test berechnen", key=f"{asset_key}_qlr"):
            with st.spinner("Berechne..."):
                qlr_res = qlr_test(close)
            if "fehler" not in qlr_res:
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    info = {k: v for k, v in qlr_res.items()
                            if k not in ("f_stats","breakpoints")}
                    st.dataframe(pd.DataFrame([info]).T.rename(columns={0:"Ergebnis"}),
                                 use_container_width=True)
                with col_q2:
                    fig_qlr = go.Figure()
                    fig_qlr.add_trace(go.Scatter(
                        x=qlr_res["breakpoints"], y=qlr_res["f_stats"],
                        line=dict(color=farbe, width=2), name="QLR F-Statistik"))
                    fig_qlr.add_hline(y=7.17, line_dash="dash", line_color=T["dn"],
                                      annotation_text="Krit. 10%: 7.17")
                    fig_qlr.update_layout(**base_layout(title="QLR Teststatistik", height=260))
                    st.plotly_chart(fig_qlr, use_container_width=True)
                st.info(qlr_res.get("Interpretation",""))

    st.divider()

    # ── 3. Stationaritätstest ─────────────────────────────────────────────────
    st.subheader("3 — Stationaritätstest & Transformation")
    st.markdown(
        "ADF und KPSS komplementär:\n"
        "- **ADF**: H₀ = Einheitswurzel (nicht stationär)\n"
        "- **KPSS**: H₀ = Stationär\n"
        "- Beide bestätigen sich gegenseitig"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ADF – Preisniveau (I(1) erwartet)**")
        st.dataframe(pd.DataFrame([adf_test(close, f"{asset_key} Close")]).T
                     .rename(columns={0:"Ergebnis"}), use_container_width=True)
    with col2:
        st.markdown("**ADF – Log-Returns (I(0) erwartet)**")
        st.dataframe(pd.DataFrame([adf_test(log_ret, f"{asset_key} Log-Returns")]).T
                     .rename(columns={0:"Ergebnis"}), use_container_width=True)

    st.markdown("**KPSS – Log-Returns (Gegenprobe)**")
    st.dataframe(pd.DataFrame([kpss_test(log_ret, f"{asset_key} Log-Returns")]).T
                 .rename(columns={0:"Ergebnis"}), use_container_width=True)

    st.info(
        f"**Transformation**: Log-Returns $r_t = \\ln(P_t/P_{{t-1}})$ "
        f"transformieren I(1) Finanzpreise in I(0) stationäre Reihen."
    )

    # Log-Returns Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["Date"].iloc[1:], y=log_ret*100,
        marker_color=[T["up"] if v>=0 else T["dn"] for v in log_ret],
        name="Log-Returns (%)",
        hovertemplate="%{x|%d.%m.%Y}<br>%{y:+.4f}%<extra></extra>",
    ))
    fig2.add_hline(y=0, line_width=1, line_dash="dot", line_color=T["border"])
    fig2.update_layout(**base_layout(
        title=f"{asset_key} Log-Returns – stationäre Zeitreihe",
        yaxis_title="Log-Return (%)", height=280, hovermode="x"))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 4. Box-Jenkins Pipeline ───────────────────────────────────────────────
    st.subheader("⚙️ 4. Box-Jenkins ARIMA Pipeline (Schritte 3–7)")

    st.markdown(
        f"**Kandidatenmodelle** (aus ACF/PACF + Theorie):"
    )
    kand_df = pd.DataFrame(
        [(o, n, b) for o,n,b in KANDIDATEN],
        columns=["Order", "Modell", "Begründung"]
    )
    st.dataframe(kand_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Train/Test Split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}% · "
        f"Zeitreihen-konform (kein Shuffling)"
    )

    if st.button(f"Box-Jenkins Pipeline starten", key=f"{asset_key}_pipeline"):
        with st.spinner("Analyse läuft (~30–60 Sek)..."):
            res = box_jenkins_pipeline(close, asset_key, FORECAST_STEPS, TRAIN_RATIO)

        if "fehler" in res:
            st.error(res["fehler"])
            return

        tt = res["train_test"]
        st.info(f"Train: {tt['n_train']} | Test: {tt['n_test']} | Split: {tt['ratio']}")

        # ADF/KPSS Diskrepanz-Erklärung (Notebook-Insight)
        if res["schritt1_integration"].get("diskrepanz"):
            st.warning(res["schritt1_integration"]["diskrepanz"])

        # Strukturbrüche auf diff. Reihe
        if res.get("strukturbrueche_diff"):
            st.markdown("**Chow-Tests auf differenzierter Reihe (Ergebnisse):**")
            for label, r in res["strukturbrueche_diff"].items():
                if "fehler" not in r:
                    st.write(f"• **{label}**: F={r.get('F-Statistik','–')}, "
                             f"p={r.get('p-Wert','–')} → {r.get('Strukturbruch','–')}")

        # ── Schritt 3: ACF/PACF ────────────────────────────────────────────────
        st.markdown("### Schritt 3: ACF & PACF")
        acf_v  = res["schritt3_acf_pacf"]["acf"]
        pacf_v = res["schritt3_acf_pacf"]["pacf"]
        kb     = res["schritt3_acf_pacf"]["konfidenzband"]
        lags   = list(range(len(acf_v)))

        col_a, col_p = st.columns(2)
        with col_a:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=lags, y=acf_v, name="ACF",
                                   marker_color=farbe, opacity=0.8))
            fig3.add_hline(y=kb,  line_dash="dash", line_color=T["dn"])
            fig3.add_hline(y=-kb, line_dash="dash", line_color=T["dn"])
            fig3.add_hline(y=0,   line_width=1, line_color=T["border"])
            fig3.update_layout(**base_layout(title="ACF", height=280,
                                              xaxis_title="Lag"))
            st.plotly_chart(fig3, use_container_width=True)
        with col_p:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=lags[1:], y=pacf_v[1:], name="PACF",
                                   marker_color=T["purple"], opacity=0.8))
            fig4.add_hline(y=kb,  line_dash="dash", line_color=T["dn"])
            fig4.add_hline(y=-kb, line_dash="dash", line_color=T["dn"])
            fig4.add_hline(y=0,   line_width=1, line_color=T["border"])
            fig4.update_layout(**base_layout(title="PACF", height=280,
                                              xaxis_title="Lag"))
            st.plotly_chart(fig4, use_container_width=True)
        st.caption(res["schritt3_acf_pacf"]["interpretation"])

        # ── Schritt 4: Modellselektion ─────────────────────────────────────────
        st.markdown("### Schritt 4: Modellvergleich – Kandidaten & Grid Search")

        tab_kand, tab_grid = st.tabs(["Kandidatenmodelle (BIC-sortiert)", "Grid Search (alle Kombinationen)"])
        with tab_kand:
            st.dataframe(res["schritt4_selektion"]["kandidaten_tabelle"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Kandidatenmodelle: ARMA(p,0,q) auf Log-Renditen (d=0). "
                "Log-Renditen sind I(0) — keine weitere Differenzierung nötig. "
                "BIC-Werte sind direkt mit Grid Search vergleichbar."
            )
        with tab_grid:
            st.dataframe(res["schritt4_selektion"]["grid_tabelle"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Grid Search: ARMA(p,0,q) auf Log-Renditen (d=0). "
                "Log-Renditen sind I(0) — keine weitere Differenzierung nötig."
            )

        st.success(res["schritt4_selektion"]["interpretation"])

        # Walk-Forward CV
        if "cross_validation" in res and "fehler" not in res["cross_validation"]:
            cv = res["cross_validation"]
            st.markdown("**Walk-Forward Cross-Validation:**")
            cv_df = pd.DataFrame({
                "Fold":  list(range(1, len(cv["RMSE_folds"])+1)),
                "RMSE":  cv["RMSE_folds"],
                "MAE":   cv["MAE_folds"],
            })
            col_cv1, col_cv2 = st.columns(2)
            with col_cv1:
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
            with col_cv2:
                st.metric("Ø RMSE", f"{cv['RMSE_mean']:.6f}")
                st.metric("Std RMSE", f"{cv['RMSE_std']:.6f}")

        # ── Schritt 5: Residualdiagnose (5-Check) ─────────────────────────────
        st.markdown("### Schritt 5: Residualdiagnose")
        st.caption(res["schritt5_residuen"].get("hinweis",""))

        tab_main, tab_bench = st.tabs(["Hauptmodell", "Benchmark (Random Walk)"])

        for tab, diag_key, label in [(tab_main, "hauptmodell", "Hauptmodell"),
                                      (tab_bench, "benchmark",    "Random Walk")]:
            with tab:
                diag = res["schritt5_residuen"].get(diag_key)
                if not diag:
                    st.info("Nicht verfügbar")
                    continue
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    stat_df = pd.DataFrame(
                        list(res["schritt5_residuen"]["statistiken"].items()
                             if diag_key=="hauptmodell" else
                             {"Ljung-Box p (Lag10)": diag["lb_p_lag10"],
                              "Ljung-Box p (Lag20)": diag["lb_p_lag20"],
                              "Jarque-Bera p":       diag["jb_p"]}.items()),
                        columns=["Kennzahl","Wert"]
                    )
                    st.dataframe(stat_df, use_container_width=True, hide_index=True)

                with col_r2:
                    residuen = diag["residuen"]
                    # Residuen-Plot
                    fig5 = go.Figure()
                    fig5.add_trace(go.Scatter(
                        x=list(range(len(residuen))), y=residuen.values,
                        mode="lines", line=dict(color=farbe, width=0.8), name="Residuen"))
                    fig5.add_hline(y=0, line_dash="dot", line_color=T["border"])
                    fig5.update_layout(**base_layout(title=f"Residuen – {label}", height=220))
                    st.plotly_chart(fig5, use_container_width=True)

                # QQ-Plot (wie im Notebook)
                res_vals = np.array(residuen.dropna())
                qq = scipy_stats.probplot(res_vals)
                theoretical_q = [p[0] for p in qq[0]]
                sample_q      = [p[1] for p in qq[0]]
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_q, y=sample_q, mode="markers",
                    marker=dict(color=farbe, size=3, opacity=0.6), name="QQ"))
                mn, mx = min(theoretical_q), max(theoretical_q)
                fig_qq.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines", line=dict(color=T["dn"], dash="dash"), name="Normalverteilung"))
                fig_qq.update_layout(**base_layout(
                    title=f"QQ-Plot – {label}",
                    xaxis_title="Theoretische Quantile",
                    yaxis_title="Stichproben-Quantile", height=260))
                st.plotly_chart(fig_qq, use_container_width=True)
                st.caption(diag["interpretation"])

        # ── Schritt 6: Koeffizienten ───────────────────────────────────────────
        st.markdown("### Schritt 6: t-Statistiken der Koeffizienten")
        st.dataframe(res["schritt6_koeffizienten"]["tabelle"],
                     use_container_width=True, hide_index=True)
        st.caption(res["schritt6_koeffizienten"]["interpretation"])

        # ── Benchmark-Vergleich ────────────────────────────────────────────────
        st.markdown("### Benchmark: ARIMA vs. Random Walk (Markteffizienz)")
        bm = res["benchmark_vergleich"]
        bm_df = pd.DataFrame([bm["random_walk"], bm["arima"]])
        st.dataframe(bm_df, use_container_width=True, hide_index=True)
        if bm["arima_besser_rmse"]:
            st.success(bm["interpretation"])
        else:
            st.warning(bm["interpretation"])

        # Train vs. Test RMSE — Overfitting-Check
        ttv = bm.get("train_test_vergleich", {})
        if ttv:
            st.markdown("#### Train vs. Test RMSE — Overfitting-Check")
            st.markdown(
                "Train RMSE ≈ Test RMSE → kein Overfitting. "
                "Ratio (Test/Train) nahe 1,0 = gut kalibriert."
            )
            ttv_rows = []
            for modell, vals in ttv.items():
                ttv_rows.append({
                    "Modell":      modell,
                    "Train RMSE":  vals["Train RMSE"],
                    "Test RMSE":   vals["Test RMSE"],
                    "Ratio":       vals["Ratio"],
                    "Diagnose":    vals["Diagnose"],
                })
            st.dataframe(
                pd.DataFrame(ttv_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Ratio = Test RMSE / Train RMSE. "
                "Ratio ≈ 1,0: gut kalibriert. "
                "> 1,5: Overfitting-Risiko. "
                "< 0,67: Underfitting-Risiko."
            )

        # ── Schritt 7: Prognose ────────────────────────────────────────────────
        st.markdown("### Schritt 7: Prognose mit 95%-Vorhersageintervall")
        st.markdown(r"$\hat{y}_{t+h|t} \pm 1.96 \cdot \hat{\sigma}_h$")

        if "fehler" not in res.get("schritt7_prognose", {}):
            prog    = res["schritt7_prognose"]
            mean_fc = prog["prognose_mean"]
            low_fc  = prog["konfidenz_lower"]
            high_fc = prog["konfidenz_upper"]
            fc_idx  = list(range(len(log_ret), len(log_ret)+FORECAST_STEPS))

            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=list(range(max(0,len(log_ret)-60), len(log_ret))),
                y=log_ret.iloc[-60:].values*100,
                name="Historisch", line=dict(color=farbe, width=2)))
            fig6.add_trace(go.Scatter(
                x=fc_idx, y=mean_fc.values*100,
                name="Prognose", line=dict(color=T["up"], width=2.5, dash="dash")))
            fig6.add_trace(go.Scatter(
                x=fc_idx+fc_idx[::-1],
                y=list(high_fc.values*100)+list(low_fc.values[::-1]*100),
                fill="toself", fillcolor="rgba(34,197,94,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="95%-KI"))
            fig6.add_hline(y=0, line_width=1, line_dash="dot", line_color=T["border"])
            fig6.update_layout(**base_layout(
                title=f"{asset_key} – ARIMA{res['schritt4_selektion']['bestes_order']} Prognose",
                yaxis_title="Log-Return (%)", height=380))
            st.plotly_chart(fig6, use_container_width=True)
            st.dataframe(prog["prognose_df"].round(6), use_container_width=True)
            st.success(prog["interpretation"])
