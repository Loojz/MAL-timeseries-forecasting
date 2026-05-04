"""
Streamlit application for MAL Time Series Forecasting.

Run with:
    streamlit run app.py
"""
import streamlit as st


def main():
    st.set_page_config(
        page_title="MAL Time Series Forecasting",
        page_icon="📈",
        layout="wide",
    )
    st.title("MAL Time Series Forecasting 📈")
    st.markdown(
        "Forecasting financial time series with ARIMA and ML methods. "
        "Project under construction."
    )


if __name__ == "__main__":
    main()
