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
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Set page title and layout
# Set page title, layout, and icon
st.set_page_config(
    page_title="📊 Statistical Arbitrage Strategy 🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
FASTAPI_BASE_URL = ""


# ENDPOINTS = {
#     "Index Data": f"{FASTAPI_BASE_URL}/get_market_data",
#     "Live Stock Prices": f"{FASTAPI_BASE_URL}/get_live_prices",
#     "Trading Signals": f"{FASTAPI_BASE_URL}/get_trading_signals"
# }

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

    # Initialize BigQuery client
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
st.markdown('<div class="section-header">⚙️ Select Parameters PCA</div>', unsafe_allow_html=True)

# 🎛 Interactive Parameter Selection
# ✅ Function to Fetch Data from BigQuery
def fetch_data(dataset: str, table: str):
    """Fetch data from BigQuery dataset and table"""
    query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"

    # Initialize BigQuery client
    return client.query(query).to_dataframe()  # Run query and return DataFrame

# ✅ Fetching functions for specific datasets
# ✅ Function to Fetch Data from BigQuery
def fetch_data(dataset: str, table: str):
    """Fetch data from BigQuery dataset and table"""
    query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"
    return client.query(query).to_dataframe()

# ✅ Fetching functions for specific datasets
def fetch_NASDAQ100_all_components():
    return fetch_data("NASDAQ100", "NASDAQ100_all_components")

def fetch_SP500_all_components():
    return fetch_data("SP500", "SP500_all_components")

def fetch_ftse100_all_components():
    return fetch_data("FTSE100", "FTSE100_all_components")

# ✅ Dictionary to Map User Selection to Fetching Functions
index_options = {
    "FTSE100": fetch_ftse100_all_components,
    "NASDAQ100": fetch_NASDAQ100_all_components,
    "SP500": fetch_SP500_all_components
}

# 📊 Interactive Index Selection in Streamlit
with st.form(key='form_bigquery_selection'):
    selected_index = st.selectbox("🔍 Choose an index to analyze:", list(index_options.keys()))
    time_period = st.slider("⏳ Select Time Period (days)", min_value=30, max_value=200, value=100)
    num_stocks = st.number_input("📈 Number of Stocks", min_value=10, max_value=60, value=20)

    # Submit button
    submitted = st.form_submit_button("🔍 Get Replication Portfolio Weights")

# ✅ Fetch Data and Process PCA on Submission
if submitted:
    with st.spinner(f"Fetching data for {selected_index} from BigQuery..."):
        underlying_df = index_options[selected_index]()  # Fetch the correct dataset
        time.sleep(2)

    if not underlying_df.empty:
        st.success(f"✅ Successfully loaded {selected_index} market data!")

        # 🎯 Step 1: Preprocess Data
        def preprocessing_X(df):
            """Preprocesses the stock price data to log returns."""
            df = df.set_index("date")
            df = df.apply(lambda x: np.log(x) - np.log(x.shift(1)))  # Log returns
            return df.dropna()

        processed_df = preprocessing_X(underlying_df)

        # 🎯 Step 2: Apply Rolling PCA & Get Stock Weights
        def rolling_pca_weights(X_log, n_stocks, window, n_pcs):
            """Computes rolling PCA and returns a DataFrame with the final stock weights."""
            tickers = X_log.columns  # All stock tickers
            selected_tickers = tickers[:n_stocks]  # Select the first `n_stocks`
            results = []

            # Rolling PCA Calculation
            for i in range(len(X_log) - window):
                X_window = X_log.iloc[i : i + window, :n_stocks]  # Select the rolling window
                pca = PCA(n_components=n_pcs)
                pca.fit(X_window)
                weights = pca.components_.T[:, 0]  # Select the first eigenvector
                results.append(weights)

            # Compute the final mean weight across rolling windows
            mean_weights = np.mean(results, axis=0)

            # Convert weights into a DataFrame
            weights_df = pd.DataFrame([mean_weights], columns=selected_tickers)
            return weights_df

        # 🎯 Step 3: Compute PCA Weights
        rep_pf = rolling_pca_weights(processed_df, num_stocks, calibration_days, n_pcs=3)

        # ✅ Display Results
        st.success("🎯 PCA Calculation Complete! Below are the weights for the selected stocks.")
        st.dataframe(rep_pf.style.format("{:.4f}"))  # Display with formatting

    else:
        st.error("🚨 No data available for this index. Try again.")


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
st.subheader("📢 Animated Strategy Description")
st.write("""
- Our trading strategy is built on a **Replication Portfolio** derived from **PCA (Principal Component Analysis)**
- We calculate the **Z-score** of the spread between the log returns of the replication portfolio and the market index
- The Z-score determines our trade signals and positions:
    - **Z-score < -2:** Go **long the index** and **short the replication portfolio**
    - **Close the position:** When the Z-score rises above -0.5
    - **Z-score > 2:** Go **short the index** and **long the replication portfolio**
    - **Close the position:** When the Z-score drops below 0.5
- In case of no trading signal generated by the model, we hold the market index to maintain market exposure
- Z-score thresholds can be amended to optimize trading opportunities
""")

# 🎯 Animated Section Header
st.markdown('<div class="section-header">⚙️ Select Z-Score Parameters</div>', unsafe_allow_html=True)

# 📊 Interactive Parameter Selection
with st.form(key='form_zscore_selection'):
    # 🎛 Slider for Calibration Days
    calibration_days = st.slider(
        "📅 Select Number of Calibration Days (Z-score calculation)",
        min_value=30, max_value=90, value=60
    )

    # 🎛 Radio Buttons for Z-Score Thresholds
    zscore_thresholds = st.radio(
        "📈 Select Z-Score Thresholds for Entering a Trade:",
        options=[
            (-2, 2),  # Option 1: -2 and 2
            (-1.5, 1.5)  # Option 2: -1.5 and 1.5
        ],
        index=0  # Default to (-2, 2)
    )

    # Fixed Threshold Information
    st.markdown("""
    - 🚨 **Note:** Positions will always close when the Z-score rises above -0.5 or falls below 0.5
    """, unsafe_allow_html=True)

    # Submit Button
    submitted = st.form_submit_button("✅ Confirm Parameters")

# ✅ Display Selected Parameters After Submission
if submitted:
    with st.spinner("Processing your selection..."):
        time.sleep(1)  # Simulating processing time
    st.success(f"🎯 Calibration Days: {calibration_days}")
    st.success(f"🎯 Z-Score Entry Thresholds: {zscore_thresholds[0]} and {zscore_thresholds[1]}")
    st.success("🎯 Position Exit Thresholds: Always fixed at -0.5 and 0.5")

    # Optionally: Display next steps or instructions
    st.markdown("""
        The selected parameters are ready to be applied to your trading strategy.
        Adjust calibration days and entry thresholds dynamically to find optimal performance!
    """)

# Simulated Strategy Output Graph
st.subheader("Strategy Output Graph")
output_data = pd.DataFrame(np.cumsum(np.random.randn(100)), columns=["Cumulative Returns"])
fig_output = px.line(output_data, y="Cumulative Returns", title="Simulated Performance of Trading Strategy versus Market Index")
st.plotly_chart(fig_output)

# 📊 Key Findings
st.subheader("📊 Key Findings")
st.write("""
- Statistical Arbitrage leverages inefficiencies in market pricing to identify profitable opportunities
- PCA (Principal Component Analysis) identifies a **replication portfolio** composed of stocks that most explain the variability in the market index. This replication portfolio serves as the foundation for our trading strategy
- By calculating the spread between the log returns of the replication portfolio and the index, trade signals can be generated to exploit potential arbitrage opportunities
- Performance is influenced by factors such as the choice of index, PCA input parameters, and threshold tuning
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



#### window_pca: Number of days the pca is calculated over to replace in streamlit



##### calibration day another place

####bt_result=z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, thresholds, dynamic=False)
### dynamics should be true in streamlit
#### input cal_ days, trade_days, thresholds,
####
####
####calibration_days = st.number_input("📅 Calibration Days", min_value=30, max_value=60, value=45)
