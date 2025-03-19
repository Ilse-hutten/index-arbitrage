import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards
import time
import requests
from google.cloud import bigquery

FASTAPI_BASE_URL = ""


# ENDPOINTS = {
#     "Index Data": f"{FASTAPI_BASE_URL}/get_market_data",
#     "Live Stock Prices": f"{FASTAPI_BASE_URL}/get_live_prices",
#     "Trading Signals": f"{FASTAPI_BASE_URL}/get_trading_signals"
# }


# Set page title and layout
# Set page title, layout, and icon
st.set_page_config(
    page_title="📊 Statistical Arbitrage Strategy 🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations & styling
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

        .subtext {
            text-align: center;
            font-size: 1.3em;
            color: #cccccc;
            animation: fadeIn 3s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Display animated title
st.markdown('<div class="title">📊 Statistical Arbitrage Strategy 🚀</div>', unsafe_allow_html=True)

# Custom CSS for animations & styling
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0px); }
        }

        .subheader {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            background: linear-gradient(90deg, #ff8c00, #ff0080);
            padding: 10px;
            border-radius: 10px;
            animation: fadeIn 1s ease-in-out;
        }

        .selectbox-container {
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Animated Section Title
st.markdown('<div class="subheader">📈 Select Your Market Index</div>', unsafe_allow_html=True)

@st.cache_data
# ✅ Function to Fetch Data from BigQuery and Standardize Column Names
def fetch_data(dataset: str, table: str, index_name: str):
    """Fetch data from BigQuery and rename the selected index column to 'price'"""
    query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"

    client = bigquery.Client()  # Initialize BigQuery client
    df = client.query(query).to_dataframe()

    # ✅ Rename the specific index column to "price"
    df.rename(columns={index_name: "price"}, inplace=True)

    return df  # Return cleaned DataFrame

# ✅ Fetching functions for specific datasets in BigQuery
def fetch_NASDAQ100_index():
    return fetch_data("NASDAQ100", "NASDAQ100_index", "NASDAQ100")

def fetch_SP500_index():
    return fetch_data("SP500", "SP500_index", "SP500")

def fetch_ftse100_index():
    return fetch_data("FTSE100", "FTSE100_index", "FTSE100")

# ✅ Dictionary to Map User Selection to BigQuery Functions
index_options = {
    "FTSE100": fetch_ftse100_index,
    "NASDAQ100": fetch_NASDAQ100_index,
    "SP500": fetch_SP500_index
}

# 📊 Interactive Index Selection in Streamlit
with st.form(key='params_for_bigquery'):
    selected_index = st.selectbox("🔍 Choose an index to analyze:", list(index_options.keys()))

    # Submit button (must be inside the form)
    submitted = st.form_submit_button("🔍 Get Market Insights")

    if submitted:
        # 🚀 Loading animation
        with st.spinner(f"Fetching data for {selected_index} from BigQuery..."):
            bigquery_data = index_options[selected_index]()  # Fetch BigQuery data dynamically
            time.sleep(2)  # Simulate loading time

        st.success(f"✅ You selected: **{selected_index}**! Let's analyze.")

        if not bigquery_data.empty:
            # 📊 Plot BigQuery market data
            fig = px.line(bigquery_data, y="price", x="date", title=f"{selected_index} Market Trend")
            st.plotly_chart(fig)
        else:
            st.error("🚨 No data available for this index. Try again.")



################################################################################################################################################
######### --> make connection with API to call the graph from the index
################################################################################################################################################

# Custom CSS for animation and styling
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0px); }
        }

        .intro-container {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            border-radius: 10px;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
            animation: fadeIn 1.5s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }

        .emoji {
            font-size: 2em;
            padding-right: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Animated Intro Text
st.markdown('<div class="intro-container">🚀 Let’s start the <span class="emoji">📊</span> Arbitrage Strategy! <br> First, select the following variables to begin.</div>', unsafe_allow_html=True)

# Custom CSS for animation and styling
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0px); }
        }

        .section-header {
            text-align: center;
            padding: 15px;
            background: linear-gradient(90deg, #ff8c00, #ff0080);
            border-radius: 10px;
            color: white;
            font-size: 1.8em;
            font-weight: bold;
            animation: fadeIn 1s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }

        .input-container {
            padding: 15px;
            animation: fadeIn 1.5s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Animated Section Header
st.markdown('<div class="section-header">⚙️ Select Strategy Parameters</div>', unsafe_allow_html=True)

# 🎛 Interactive Parameter Selection
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    time_period = st.slider(
        "⏳ Select Time Period (days)",
        min_value=30,
        max_value=365,
        value=180,
        help="Adjust the time period for analyzing market trends."
    )

    calibration_days = st.number_input(
        "📅 Calibration Days",
        min_value=30,
        max_value=60,
        value=45,
        help="Choose the number of days to calibrate the strategy."
    )

    num_stocks = st.number_input(
        "📈 Number of Stocks",
        min_value=10,
        max_value=60,
        value=20,
        help="Define the number of stocks to include in the arbitrage strategy."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Section: Trading Strategy
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0px); }
        }

        .strategy-header {
            text-align: center;
            padding: 15px;
            background: linear-gradient(90deg, #00b09b, #96c93d);
            border-radius: 10px;
            color: white;
            font-size: 1.8em;
            font-weight: bold;
            animation: fadeIn 1s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }

        .strategy-description {
            font-size: 1.2em;
            text-align: center;
            color: #444444;
            padding: 10px;
            animation: fadeIn 1.5s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Animated Section Header
st.markdown('<div class="strategy-header">📊 Trading Strategy Execution</div>', unsafe_allow_html=True)

# 📢 Animated Strategy Description
st.markdown('<div class="strategy-description">🚀 The system will leverage <b>PCA (Principal Component Analysis)</b> to identify high-probability arbitrage opportunities, optimizing stock selection dynamically.</div>', unsafe_allow_html=True)

# Simulated Strategy Output Graph
st.subheader("Strategy Output Graph")
output_data = pd.DataFrame(np.cumsum(np.random.randn(100)), columns=["Cumulative Returns"])
fig_output = px.line(output_data, y="Cumulative Returns", title="Simulated Trading Performance")
st.plotly_chart(fig_output)

# Section: Summary of Findings
st.subheader("📊 Key Findings")
st.write("""
- Statistical Arbitrage identifies inefficiencies in market pricing.
- PCA helps in reducing dimensionality and finding key trading signals.
- Performance varies based on index and input parameters.
- Diversification across multiple stocks improves risk-adjusted returns.
""")

# Section: Download Strategy
st.subheader("📥 Download Your Strategy")
if st.button("Download Strategy as CSV"):
    strategy_data = pd.DataFrame({
        "Parameter": ["Time Period", "Calibration Days", "Number of Stocks"],
        "Value": [time_period, calibration_days, num_stocks]
    })
    strategy_data.to_csv("strategy_output.csv", index=False)
    st.success("Your strategy has been downloaded!")

# Final Note
st.info("💡 *'Just holding might be the better method if you want to keep it simple.'*")


# st.title("Stat Arb!")
# st.write("Choose your index")
# st.write("FTSE100      NASDAQ100      SP500")
# st.write("Some kind of graph")
# st.write("choose your variables")
# st.write("Data source: google or local")
# st.write("PCA input: Time periood, calibratioon days, number of stocks")
# st.write("Trading strategy")
# st.write("Graph for the output")
# st.write("Summary of key findings")
# st.write("Download your strategy")
# st.write("Just hold is the bettter method if you want to keep it simple")
