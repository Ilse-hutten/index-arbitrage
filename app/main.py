import os
import pandas as pd
from frame import Frame
from testing import z_score_trading
from google.cloud import storage
from google.cloud import bigquery

data=Frame()
stock_price=data.dataset()
#
# #importing FTSE 100 data from Google cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../data/" + 'lewagon-statistical-arbitrage-ae470f7dcd48.json'
client = storage.Client()
bucket = client.get_bucket('stat_arb')
# #
# Set your GCP project details
project_id = "lewagon-statistical-arbitrage"
dataset_id = "FTSE_100_main"  # Replace with your dataset name
table_name = "FTSE100_csv"

# Construct full table path
table_id = f"{project_id}.{dataset_id}.{table_name}"
query = "SELECT * FROM `lewagon-statistical-arbitrage.FTSE_100_main.FTSE100_csv` ORDER BY Unnamed_0 ASC"
client = bigquery.Client(project=project_id)
target = client.query(query).to_dataframe()

target.rename(columns={'Unnamed_0': 'Date',"close": "FTSE price"}, inplace=True)
target.set_index('Date', inplace=True)
target_close_price = pd.DataFrame(target["FTSE price"])

# #PCA function

# Specifically for FTSE data!!! Needs to be adjusted for other dataframes
#
os.chdir('..')
cwd=os.getcwd()

daily_weight = pd.read_csv(cwd + "/data/daily_weights.csv")
#daily_weight = daily_weight.rename(columns = lambda x : str(x)[:-2])
daily_weight = daily_weight.rename(columns={'Date': 'date'})
daily_weight["date"] = pd.to_datetime(daily_weight["date"])

# #z-score trading simulation
# #
pca_weights_df=daily_weight
underlying_df=stock_price
target_df=target_close_price
cal_days=60
trade_days=30

bt_result=z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, dynamic=False)
bt_result.to_csv(cwd + "/data/backtesting.csv")
