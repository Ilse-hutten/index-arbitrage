#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[3]:


import sys
import os
sys.path.append(os.path.abspath("../app"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/sb/lewagon_london/project_lewagon/stock-stat-replica/data/lewagon-statistical-arbitrage-ae470f7dcd48.json'


# In[4]:


from data_query import fetch_NASDAQ100_index
from data_query import fetch_NASDAQ100_all_components
from data_query import fetch_SP500_index
from data_query import fetch_SP500_all_components
from data_query import fetch_ftse100_index
from data_query import fetch_ftse100_all_components
from PCA_function import rolling_pca_weights
from preprocessing import preprocessing_X
from sklearn.decomposition import PCA
from typing import List


# In[18]:


index_selected='FTSE100'
if index_selected=='NASDAQ100':
    target_df= fetch_NASDAQ100_index()
    underlying_df=fetch_NASDAQ100_all_components()
elif index_selected=='SP500':
    target_df= fetch_SP500_index()
    underlying_df=fetch_SP500_all_components()
elif index_selected=='FTSE100':
    target_df= fetch_ftse100_index()
    underlying_df=fetch_ftse100_all_components()

n_stocks = 30               # number of stocks used for the replication
window = 30                 # period the trading strat goes
n_pcs = 3   
if 'date' in underlying_df.columns and not underlying_df['date'].empty:
    underlying_df.set_index('date', inplace=True)

# Create log returns to remove stationarity
log_returns = np.log(underlying_df / underlying_df.shift(1))

# Drop NaN values caused by the shift
log_returns = log_returns.dropna()
X_log = log_returns.copy()
stock_log_features = X_log.columns

# Scaling data
scaler = StandardScaler()
scaler.fit(X_log)
X_log = pd.DataFrame(scaler.transform(X_log), columns=stock_log_features, index=log_returns.index)

rep_pf = rolling_pca_weights(X_log, n_stocks, window, n_pcs)


# In[19]:


daily_weights_df=rep_pf.copy()
daily_weights_df.index=pd.to_datetime(daily_weights_df.index)
daily_weights_df.index


# In[20]:


target_df.set_index('date',inplace=True)


# In[21]:


df=target_df
df


# In[22]:


# SPREAD CODE:

pca_date = '2024-03-18'

# Filter stocks with values > 0 on the specific date
filtered_columns = daily_weights_df.loc[pca_date][daily_weights_df.loc[pca_date] > 0].index
# replication pf weights taken from output function
rep_pf = daily_weights_df.loc[[pca_date], filtered_columns]

# log returns of rep pf
rep_pf_log_returns_daily = log_returns[rep_pf.columns]

# log return times weights (results)
rep_pf_results = rep_pf_log_returns_daily.mul(rep_pf.iloc[0], axis=1)
rep_pf_results["total_rep_pf"] = rep_pf_results.sum(axis=1)
rep_pf_results

# Calculate log returns FTSE100 / Index    
FTSE_log_return = np.log(df["FTSE100"] / df["FTSE100"].shift(1)).dropna()
FTSE_log_return

# DF of spread
spread_df = pd.DataFrame()  # Create a new DataFrame for the spread
spread_df["spread"] = FTSE_log_return - rep_pf_results["total_rep_pf"]  # Add a column named "spread"
training_spread_df = spread_df.loc[:pca_date].iloc[:-1]  # Data up to the PCA date, excluding the date itself


# In[23]:


training_spread_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler


spread_values = spread_values = training_spread_df
df=training_spread_df

train_size = int(len(spread_values) * 0.8)
train_spread, test_spread = spread_values[:train_size], spread_values[train_size:]

scaler = StandardScaler()
train_spread_scaled = scaler.fit_transform(train_spread)
test_spread_scaled = scaler.transform(test_spread)

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 30
X_train, y_train = create_sequences(train_spread_scaled, sequence_length)
X_test, y_test = create_sequences(test_spread_scaled, sequence_length)


model = Sequential()
model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Forecast future values
future_predictions = []
last_sequence = test_spread_scaled[-sequence_length:].reshape(1, sequence_length, 1)

for _ in range(30):  # Predict next 30 days
    pred = model.predict(last_sequence)[0, 0]
    future_predictions.append(pred)
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = pred

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates (skip weekends)
future_dates = pd.date_range(df.index[-1], periods=30, freq="B")  # 'B' for business days

# Generate Buy/Sell signals (Simple: Buy if increasing, Sell if decreasing)
buy_signals, sell_signals = [], []
for i in range(1, len(future_predictions)):
    if future_predictions[i] > future_predictions[i-1]:  # Upward trend
        buy_signals.append((future_dates[i], future_predictions[i, 0]))
    else:  # Downward trend
        sell_signals.append((future_dates[i], future_predictions[i, 0]))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-100:], df["spread"].values[-100:], label="Historical Spread", color="blue")
plt.plot(future_dates, future_predictions, label="Forecasted Spread", color="orange", linestyle="dashed")
plt.scatter(*zip(*buy_signals), color="green", marker="^", label="Buy Signal")
plt.scatter(*zip(*sell_signals), color="red", marker="v", label="Sell Signal")

plt.xlabel("Date")
plt.ylabel("Spread Value")
plt.title("Spread Forecast with Buy/Sell Signals")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




