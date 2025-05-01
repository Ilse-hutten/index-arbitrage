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
from output import alternative_asset_return
import datetime
from PCA_function import rolling_pca_weights
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()  # Automatically loads variables from the .env file
import os
from google.oauth2 import service_account
from google.cloud import bigquery

st.set_page_config(
    page_title="📊 Statistical Arbitrage Strategy 🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Build credentials from environment variables
# Read private key from environment variables (set in Railway)
raw_private_key = os.getenv("private_key")

if raw_private_key:
    # Replace the literal "\n" with actual newlines
    private_key = raw_private_key.replace("\\n", "\n")
else:
    raise ValueError("Missing private key from environment")


credentials_info = {
    "type": "service_account",
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    "private_key": private_key,
    "client_email": os.getenv("client_email"),
    "client_id": os.getenv("client_id"),
    "auth_uri": os.getenv("auth_uri"),
    "token_uri": os.getenv("token_uri"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_cert_url"),
    "client_x509_cert_url": os.getenv("client_cert_url"),
}
credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = bigquery.Client(credentials=credentials)

URL = "https://index-arbitrage-fqciuj24i2dwjmbgaytvhl.streamlit.app/"
#URL = "https://developers-254643980168.europe-west1.run.app/fetch_btresult_rolling_pca"
# URL = "http://127.0.0.1:8000/fetch_btresult_rolling_pca"

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

def fetch_CRYPTO_index():
    return fetch_data("CRYPTO", "CRYPTO_INDEX")

# ✅ Dictionary to Map User Selection to BigQuery Functions
index_options = {
    "FTSE100": fetch_ftse100_index,
    "NASDAQ100": fetch_NASDAQ100_index,
    "SP500": fetch_SP500_index,
    "CRYPTO": fetch_CRYPTO_index
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

def fetch_CRYPTO_all_components():
    return fetch_data("CRYPTO", "CRYPTO_ALL_COMPONENTS")

# ✅ Dictionary to Map User Selection to Fetching Functions
index_options = {
    "FTSE100": fetch_ftse100_all_components,
    "NASDAQ100": fetch_NASDAQ100_all_components,
    "SP500": fetch_SP500_all_components,
    "CRYPTO":fetch_CRYPTO_all_components
}

# 📊 Interactive Index Selection in Streamlit
with st.form(key='form_bigquery_selection'):
    selected_index = st.selectbox("🔍 Choose an index to analyze:", list(index_options.keys()))
    pca_date = st.date_input("📅 Select Date for PCA Analysis", value=datetime.date(2025, 1, 14))  # Interactive date input
    num_stocks = st.number_input("📈 Number of Stocks", min_value=10, max_value=60, value=20)
    windows = st.slider("📅 Select Number of Calibration Days (PCA Window)", min_value=30, max_value=90, value=60)
    n_pcs = st.slider("🧮 Select Number of Principal Components (n_pcs)", min_value=1, max_value=10, value=3)

    # Submit button
    submitted = st.form_submit_button("🔍 Get Replication Portfolio Weights")

# ✅ Fetch Data and Process PCA on Submission
if submitted:
    # st.write(requests.get(URL, params={"cal_days":60, "trade_days":30,"n_stocks":num_stocks,"window":windows,"n_pcs":n_pcs,"index_selected":selected_index}).json())

    rep_pf = pd.DataFrame(requests.get(URL, params={"cal_days":60, "trade_days":30,"n_stocks":num_stocks,"window":windows,"n_pcs":n_pcs,"index_selected":selected_index}).json()["rep_pf"])
    st.write(pd.DataFrame(rep_pf))
    pca_date_str = str(pca_date)  # Convert Streamlit date input to string
    pca_date = pd.to_datetime(pca_date_str)  # Ensure it's a datetime object
    rep_pf.set_index('date', inplace=True)
    rep_pf.index = pd.to_datetime(rep_pf.index)  # Ensure index is in datetime format
    if pca_date in rep_pf.index:
        rep_pf_for_date = rep_pf.loc[[pca_date]]  # Get weights for the specific date

        # Filter to only include stocks with weights > 0
        filtered_rep_pf_for_date = rep_pf_for_date.loc[:, rep_pf_for_date.iloc[0] > 0]

        # Format the date row index to display only the date (no time)
        filtered_rep_pf_for_date.index = filtered_rep_pf_for_date.index.strftime('%Y-%m-%d')

        # ✅ Display Results
        st.success("🎯 PCA Calculation Complete! Below are the weights for the selected stocks for the day you choose.")
        st.dataframe(filtered_rep_pf_for_date.style.format("{:.4f}"))  # Format DataFrame for clear output
    else:
        st.error(f"🚨 Selected date {pca_date.date()} not found in the PCA weights. Try adjusting the PCA window.")

    df_numeric = filtered_rep_pf_for_date.drop(columns=["date"], errors="ignore").T
    df_numeric.columns = ["Stock Value"]  # Rename the column for clarity
    df_numeric.reset_index(inplace=True)  # Convert index to a column
    df_numeric.rename(columns={"index": "Stock Symbol"}, inplace=True)

    # Create an interactive bar chart with Plotly
    fig = px.bar(
        df_numeric,
        x="Stock Symbol",
        y="Stock Value",
        title="Stock Values Bar Chart",
        labels={"Stock Value": "Value", "Stock Symbol": "Stock"},
        color="Stock Value",  # Optional: Color bars based on value
        text="Stock Value",   # Display values on bars
    )

    # Update layout for better readability
    fig.update_traces(texttemplate='%{text:.4f}', textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels

    # Display the plot in Streamlit
    st.plotly_chart(fig)


#     with st.spinner(f"Fetching data for {selected_index} from BigQuery..."):
#         # Fetch the correct dataset
#         underlying_df = index_options[selected_index]()
#         time.sleep(2)

#     if not underlying_df.empty:
#         st.success(f"✅ Successfully loaded {selected_index} market data!")

#         # 🎯 Step 1: Preprocess Data
#         def preprocessing_X(df):
#             """Preprocesses the stock price data to log returns."""
#             df = df.set_index("date")
#             df.index = pd.to_datetime(df.index)  # Ensure index is in datetime format
#             df = df.apply(lambda x: np.log(x) - np.log(x.shift(1)))  # Log returns
#             return df.dropna()

#         processed_df = preprocessing_X(underlying_df)

#         # Ensure `pca_date` is a valid datetime object
#         pca_date_str = str(pca_date)  # Convert Streamlit date input to string
#         pca_date = pd.to_datetime(pca_date_str)  # Ensure it's a datetime object

#         # 🎯 Step 2: Apply Rolling PCA for the Selected Date
#         if pca_date in processed_df.index:
#             # Use the existing `rolling_pca_weights` function
#             rep_pf = rolling_pca_weights(
#                 X_log=processed_df,           # Log returns DataFrame
#                 n_stocks=num_stocks,          # Number of stocks
#                 window_pca=calibration_days,  # PCA window (selected via slider)
#                 n_pcs=n_pcs                   # Number of principal components (selected via slider)
#             )

#             # Filter weights for the selected PCA date
#             if pca_date in rep_pf.index:
#                 rep_pf_for_date = rep_pf.loc[[pca_date]]  # Get weights for the specific date

#                 # Filter to only include stocks with weights > 0
#                 filtered_rep_pf_for_date = rep_pf_for_date.loc[:, rep_pf_for_date.iloc[0] > 0]

#                 # Format the date row index to display only the date (no time)
#                 filtered_rep_pf_for_date.index = filtered_rep_pf_for_date.index.strftime('%Y-%m-%d')

#                 # ✅ Display Results
#                 st.success("🎯 PCA Calculation Complete! Below are the weights for the selected stocks.")
#                 st.dataframe(filtered_rep_pf_for_date.style.format("{:.4f}"))  # Format DataFrame for clear output
#             else:
#                 st.error(f"🚨 Selected date {pca_date.date()} not found in the PCA weights. Try adjusting the PCA window.")
#         else:
#             st.error(f"🚨 Selected date {pca_date.date()} not found in the dataset. Try another date.")
#     else:
#         st.error("🚨 No data available for this index. Try again.")


# # st.write(filtered_rep_pf_for_date.T)
# # Section: Trading Strategy
# st.markdown("""
#     <style>
#         @keyframes fadeIn {
#             from { opacity: 0; transform: translateY(-10px); }
#             to { opacity: 1; transform: translateY(0px); }
#         }

#         .strategy-header {
#             text-align: center;
#             padding: 15px;
#             background: linear-gradient(90deg, #00b09b, #96c93d);
#             border-radius: 10px;
#             color: white;
#             font-size: 1.8em;
#             font-weight: bold;
#             animation: fadeIn 1s ease-in-out;
#             box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
#         }

#         .strategy-description {
#             font-size: 1.2em;
#             text-align: center;
#             color: #444444;
#             padding: 10px;
#             animation: fadeIn 1.5s ease-in-out;
#         }
#     </style>
# """, unsafe_allow_html=True)

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


# 📊 Interactive Parameter Selection
with st.form(key='form_zscore_selection'):
    # 🎛 Slider for Calibration Days
    calibration_days = st.slider(
        "📅 Select Number of Calibration Days (Z-score calculation)",
        min_value=30, max_value=90, value=60
    )

    # # 🎛 Radio Buttons for Z-Score Thresholds
    # zscore_thresholds = st.radio(
    #     "📈 Select Z-Score Thresholds for Entering a Trade:",
    #     options=[
    #         (-2, 2),  # Option 1: -2 and 2
    #         (-1.5, 1.5)  # Option 2: -1.5 and 1.5
    #     ],
    #     index=0  # Default to (-2, 2)
    # )
    # zscore_thresholds = list(zscore_thresholds)
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
    result = requests.get(URL, params={"cal_days":calibration_days, "trade_days":30,"n_stocks":num_stocks,"window":windows,"n_pcs":n_pcs,"index_selected":selected_index})
    bt_result = pd.DataFrame(result.json()["bt_result"])

    st.success(f"🎯 Calibration Days: {calibration_days}")
    #st.success(f"🎯 Z-Score Entry Thresholds: {zscore_thresholds[0]} and {zscore_thresholds[1]}")
    #st.success("🎯 Position Exit Thresholds: Always fixed at -0.5 and 0.5")
    #st.dataframe(bt_result)
    # Optionally: Display next steps or instructions
    st.markdown("""
        The selected parameters are ready to be applied to your trading strategy.
        Adjust calibration days and entry thresholds dynamically to find optimal performance!
    """)

    st_output = alternative_asset_return(bt_result)
    st_output["Total Strategy return"] = np.log(st_output["strategy"]/st_output["strategy"].shift(1))
    st.dataframe(st_output)

    #Simulated Strategy Output Graph
    st.subheader("Strategy Output Graph")
    output_data = pd.DataFrame(st_output)
    # Convert to DataFrame and reset index
    #output_data = pd.DataFrame(st_output).reset_index()  # Ensures index is a column
    # Create line plot with two lines
    fig_output = px.line(output_data, x="index",
                     y=["target entry", "strategy"],
                     title="Simulated Performance: Target Entry vs Strategy",
                     labels={"index": "Time", "value": "Returns"},
                     color_discrete_map={"Target Entry": "blue", "Strategy": "red"})  # Custom colors

    # Display the plot
    st.plotly_chart(fig_output)

    # Count occurrences of -1 and 1 in the "direction" column
    count_negative_one = (st_output["direction"] == -1).sum()
    count_positive_one = (st_output["direction"] == 1).sum()

    # Sum excess return and daily target return
    total_strtategy_return = st_output["Total Strategy return"].sum()*100
    total_daily_target_return = st_output["daily target return"].sum()*100
    total_excess_return = st_output["excess return"].sum()*100

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        "Metric": ["Count of -1", "Count of 1", "Replication Strategy", "Target index Return", "Excess Return"],
        "Value": [count_negative_one, count_positive_one, total_strtategy_return, total_daily_target_return,total_excess_return]
    })

    # Title
    st.title("📊 Strategy Metrics Dashboard")

    # Add some styling
    st.markdown(
        """
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
            text-align: center;
        }
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
        """,
        unsafe_allow_html=True
    )

    # Display metrics in nice format
    st.markdown('<p class="big-font">🚀 Summary Metrics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.markdown(f'<div class="metric-box">📉 <b>Number of Short Trades </b><br>{count_negative_one}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-box">📈 <b>Number of Long Trades</b><br>{count_positive_one}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-box">💰 <b>Our Strategy Return %</b><br>{total_strtategy_return:.4f}</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-box">📊 <b>Total Daily Target Return %</b><br>{total_daily_target_return:.4f}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-box">📊 <b>Total Excess Return %</b><br>{total_excess_return:.4f}</div>', unsafe_allow_html=True)

    # # Display DataFrame in a nice table format
    # st.markdown("### 📋 Detailed Summary Table")
    # st.dataframe(summary_df.style.format({"Value": "{:.4f}"}))


# 📊 Key Findings
# st.subheader("📊 Key Findings")
# st.write("""
# - Statistical Arbitrage leverages inefficiencies in market pricing to identify profitable opportunities
# - PCA (Principal Component Analysis) identifies a **replication portfolio** composed of stocks that most explain the variability in the market index. This replication portfolio serves as the foundation for our trading strategy
# - By calculating the spread between the log returns of the replication portfolio and the index, trade signals can be generated to exploit potential arbitrage opportunities
# - Performance is influenced by factors such as the choice of index, PCA input parameters, and threshold tuning
# """)

# # Final Note
# st.info("💡 *'Just holding might be the better method if you want to keep it simple.'*")


####bt_result=z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, thresholds, dynamic=False)
### dynamics should be true in streamlit
#### input cal_ days, trade_days, thresholds,
####calibration_days = st.number_input("📅 Calibration Days", min_value=30, max_value=60, value=45)
