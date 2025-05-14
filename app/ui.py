import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards
import time
import requests
from google.cloud import bigquery
from sklearn.decomposition import PCA
from .output import alternative_asset_return
import datetime
from .PCA_function import rolling_pca_weights
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

load_dotenv()

def render_app():
    st.set_page_config(
        page_title="📊 Statistical Arbitrage Strategy 🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)

    URL = "http://127.0.0.1:8000/fetch_btresult_rolling_pca"

    # --- Custom Styles ---
    st.markdown("""
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .title {
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                color: #ffffff;
                background: linear-gradient(45deg, #6a11cb, #2575fc);
                padding: 15px;
                border-radius: 10px;
                animation: fadeIn 2s ease-in-out;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">📊 Statistical Arbitrage Strategy 🚀</div>', unsafe_allow_html=True)

    # --- Fetch index data ---
    def fetch_data(dataset: str, table: str, index_name: str = None):
        query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"
        df = client.query(query).to_dataframe()
        if index_name:
            df.rename(columns={index_name: "price"}, inplace=True)
        return df

    def fetch_NASDAQ100_index(): return fetch_data("NASDAQ100", "NASDAQ100_index", "NASDAQ100")
    def fetch_SP500_index(): return fetch_data("SP500", "SP500_index", "SP500")
    def fetch_ftse100_index(): return fetch_data("FTSE100", "FTSE100_index", "FTSE100")
    def fetch_CRYPTO_index(): return fetch_data("CRYPTO", "CRYPTO_INDEX")

    index_options = {
        "FTSE100": fetch_ftse100_index,
        "NASDAQ100": fetch_NASDAQ100_index,
        "SP500": fetch_SP500_index,
        "CRYPTO": fetch_CRYPTO_index
    }

    with st.form(key='params_for_bigquery'):
        selected_index = st.selectbox("🔍 Choose an index to analyze:", list(index_options.keys()))
        submitted = st.form_submit_button("🔍 Get Market Insights")
        if submitted:
            with st.spinner(f"Fetching data for {selected_index} from BigQuery..."):
                bigquery_data = index_options[selected_index]()
                time.sleep(2)

            st.success(f"✅ You selected: **{selected_index}**! Let's analyze.")

            if not bigquery_data.empty:
                fig = px.line(bigquery_data, y="price", x="date", title=f"{selected_index} Market Trend")
                st.plotly_chart(fig)
            else:
                st.error("🚨 No data available for this index. Try again.")

    # --- Replication Portfolio PCA ---
    def fetch_all_components(dataset: str, table: str):
        query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"
        return client.query(query).to_dataframe()

    def fetch_NASDAQ100_all_components(): return fetch_all_components("NASDAQ100", "NASDAQ100_all_components")
    def fetch_SP500_all_components(): return fetch_all_components("SP500", "SP500_all_components")
    def fetch_ftse100_all_components(): return fetch_all_components("FTSE100", "FTSE100_all_components")
    def fetch_CRYPTO_all_components(): return fetch_all_components("CRYPTO", "CRYPTO_ALL_COMPONENTS")

    index_options = {
        "FTSE100": fetch_ftse100_all_components,
        "NASDAQ100": fetch_NASDAQ100_all_components,
        "SP500": fetch_SP500_all_components,
        "CRYPTO": fetch_CRYPTO_all_components
    }

    with st.form(key='form_bigquery_selection'):
        selected_index = st.selectbox("🔍 Choose an index to analyze:", list(index_options.keys()))
        pca_date = st.date_input("📅 Select Date for PCA Analysis", value=datetime.date(2025, 1, 14))
        num_stocks = st.number_input("📈 Number of Stocks", min_value=10, max_value=60, value=20)
        windows = st.slider("📅 Calibration Days (PCA)", min_value=30, max_value=90, value=60)
        n_pcs = st.slider("🧮 Principal Components (n_pcs)", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("🔍 Get Replication Portfolio Weights")

    if submitted:
        rep_pf = pd.DataFrame(requests.get(URL, params={
            "cal_days": 60,
            "trade_days": 30,
            "n_stocks": num_stocks,
            "window": windows,
            "n_pcs": n_pcs,
            "index_selected": selected_index
        }).json()["rep_pf"])

        rep_pf['date'] = pd.to_datetime(rep_pf['date'], errors='coerce').dt.date
        if pca_date in rep_pf["date"].values:
            rep_pf_for_date = rep_pf[rep_pf['date'] == pca_date].drop(columns=["date"])
            filtered_rep_pf_for_date = rep_pf_for_date.loc[:, rep_pf_for_date.iloc[0] > 0]
            st.success("🎯 PCA Calculation Complete! Below are the weights for the selected stocks.")
            st.dataframe(filtered_rep_pf_for_date.style.format("{:.4f}"))

            df_numeric = filtered_rep_pf_for_date.T.reset_index()
            df_numeric.columns = ["Stock Symbol", "Stock Value"]
            fig = px.bar(df_numeric, x="Stock Symbol", y="Stock Value", text="Stock Value")
            fig.update_traces(texttemplate='%{text:.4f}', textposition="outside")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        else:
            st.error(f"🚨 Selected date {pca_date} not found in PCA weights. Adjust the window.")

    # --- Z-Score Trading Strategy ---
    with st.form(key='form_zscore_selection'):
        calibration_days = st.slider("📅 Calibration Days (Z-score)", 30, 90, 60)
        st.markdown("- 🚨 **Note:** Positions close at Z-score = -0.5 or 0.5", unsafe_allow_html=True)
        submitted = st.form_submit_button("✅ Confirm Parameters")

    if submitted:
        with st.spinner("Processing your selection..."):
            time.sleep(1)
        result = requests.get(URL, params={
            "cal_days": calibration_days,
            "trade_days": 30,
            "n_stocks": num_stocks,
            "window": windows,
            "n_pcs": n_pcs,
            "index_selected": selected_index
        })
        bt_result = pd.DataFrame(result.json()["bt_result"])
        st_output = alternative_asset_return(bt_result)
        st_output["Total Strategy return"] = np.log(st_output["strategy"] / st_output["strategy"].shift(1))
        st.dataframe(st_output)

        st.subheader("Strategy Output Graph")
        fig_output = px.line(st_output, x="index", y=["target entry", "strategy"],
                             title="Simulated Performance: Target Index vs Strategy",
                             labels={"index": "Time", "value": "Returns"})
        st.plotly_chart(fig_output)

        count_negative_one = (st_output["direction"] == -1).sum()
        count_positive_one = (st_output["direction"] == 1).sum()
        total_strategy_return = round(st_output["Total Strategy return"].sum() * 100, 2)
        total_daily_target_return = round(st_output["daily target return"].sum() * 100, 2)

        #total_excess_return = st_output["excess return"].sum() * 100

        st.title("📊 Strategy Metrics Dashboard")
        st.markdown("""
            <style>
                .big-font { font-size:30px !important; font-weight: bold; text-align: center; }
                .metric-box {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px;
                    text-align: center;
                    font-size: 20px;
                    background-color: #f9f9f9;
                }
            </style>
        """, unsafe_allow_html=True)


        # Display metrics in nice format
        st.markdown('<p class="big-font">🚀 Summary Metrics</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.markdown(f'<div class="metric-box">📉 <b>Number of Short Trades </b><br>{count_negative_one}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box">📈 <b>Number of Long Trades</b><br>{count_positive_one}</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-box">💰 <b>Replication Strategy Return %</b><br>{total_strategy_return:.4f}</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-box">📊 <b>Index Return %</b><br>{total_daily_target_return:.4f}</div>', unsafe_allow_html=True)
       # col4.markdown(f'<div class="metric-box">📊 <b>Total Excess Return %</b><br>{total_excess_return:.4f}</div>', unsafe_allow_html=True)
