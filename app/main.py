import os
import pandas as pd
from frame import Frame
from testing import z_score_trading
from google.cloud import storage
from google.cloud import bigquery
from data_query import fetch_NASDAQ100_index
from data_query import fetch_NASDAQ100_all_components
from data_query import fetch_SP500_index
from data_query import fetch_SP500_all_components
from data_query import fetch_ftse100_index
from data_query import fetch_ftse100_all_components
from PCA_function import rolling_pca_weights
from preprocessing import preprocessing_X
from sklearn.decomposition import PCA
#
#-----Pulling data from Big Query
#
index_selected='sp500'
#
if index_selected=='nasdaq':
    target_df= fetch_NASDAQ100_index()
    underlying_df=fetch_NASDAQ100_all_components()

if index_selected=='sp500':
    target_df= fetch_SP500_index()
    underlying_df=fetch_SP500_all_components()

if index_selected=='ftse':
    target_df= fetch_ftse100_index()
    underlying_df=fetch_ftse100_all_components()
#
#-----pre-processing the components
processed_df=preprocessing_X(underlying_df)
#
#-----PCA function
#
# Define input variables
X_log=processed_df
n_stocks = 30               # number of stocks used for the replication
window = 30                 # period the trading strat goes
n_pcs = 3                   # number of eigenvectors
#
# Get weights
rep_pf = rolling_pca_weights(X_log, n_stocks, window, n_pcs)
#print(rep_pf)
#
#
#------Regression signal placeholder

# -----z-score trading simulation
#
pca_weights_df=rep_pf
#underlying_df=stock_price
#target_df=target_close_price
cal_days=60                 # number of days for the z score
trade_days=30               # maximum number of trading days
thresholds=[2,200,-2,-200]  # thresholds for trading signals
                            # [short minimum threshold, short maximum threshold, long minimum threshold, long maximum threshold]
exit_levels=[0.5,-0.5]      # thresholds for closing a trade
                            # [exit level long position, exit level short position]

#calling the simulation
bt_result=z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, thresholds, exit_levels, True)
#bt_result.to_csv(cwd + "/data/backtesting.csv")

# needs to be called to the API bt_result['spread']
