#!/usr/bin/env python
# coding: utf-8

# # PCA for PF Construction

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


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


# In[84]:
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


daily_weights_df=rep_pf.copy()
daily_weights_df

daily_weights_df.index=pd.to_datetime(daily_weights_df.index)
daily_weights_df.index

target_df.set_index('date',inplace=True)
df=target_df
df


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

training_spread_df
#result of training_spread_df
# 	spread
# date
# 2022-02-02	-0.001915
# 2022-02-03	0.005052
# 2022-02-04	0.009731
# 2022-02-07	0.005202
# 2022-02-08	-0.002083
# ...	...
# 2024-03-11	0.00242
# 2024-03-12	-0.002849
# 2024-03-13	0.005767
# 2024-03-14	0.001719
# 2024-03-15	0.026459
# 537 rows × 1 columns


def get_economic_indicators(start_date, end_date):
    tickers = [
        "^GSPC",   # S&P 500
        "^VIX",    # Volatility Index
        "^TNX",    # 10-Year Treasury Yield
        "^FVX",    # 5-Year Treasury Yield
        "GC=F",    # Gold Futures
        "CL=F",    # Crude Oil Futures
        "GBPUSD=X" # GBP/USD Exchange Rate
    ]

    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Close']
    return data


# In[128]:


economic_indicators = get_economic_indicators('2022-01-31', '2025-03-14')


# In[ ]:





# In[129]:


abstickers = {
        "^GSPC":"SP500",
        "^VIX":"VIX",
        "^TNX":"TNX",
        "^FVX":"FVX",
        "GC=F":"GF",
        "CL=F":"COF",
        "GBPUSD=X":"GBPUSD"
}


# In[ ]:





# In[6]:


from data_query  import eco_df
bg_data=eco_df()


# In[7]:


eco_df


# In[8]:


eco_df.isna().sum()


# In[133]:


eco_df.fillna(method='ffill', inplace=True)


# In[134]:


def convert_to_log_returns(data):
    log_returns = data.copy()

    # List of indicators to convert to log returns
    convert_tickers = ['SP500', 'GF', 'COF', 'GBPUSD']

    # Apply log returns conversion
    for ticker in convert_tickers:
        if ticker in log_returns.columns:
            log_returns[ticker] = np.log(log_returns[ticker] / log_returns[ticker].shift(1))

    # Drop the first row which will have NaN values due to the shift
    log_returns = log_returns.dropna()

    return log_returns

log_return_eco = convert_to_log_returns(eco_df)


# In[135]:


log_return_eco


# In[136]:


def prepare_data_for_lstm(df, economic_indicators, sequence_length=20):

    merged_df = df.join(economic_indicators, how='inner')

    # Calculate additional features
    merged_df['volatility_ftse'] = df['FTSE100'].pct_change().rolling(window=20).std()
    merged_df['volatility_pf'] = df['total_rep_pf'].pct_change().rolling(window=20).std()
    merged_df['spread_ma5'] = df['spread'].rolling(window=5).mean()
    merged_df['spread_ma20'] = df['spread'].rolling(window=20).mean()

    merged_df['rsi_spread'] = calculate_rsi(df['spread'], window=14)
    merged_df = merged_df.dropna()


    features = ['SP500', 'VIX', 'TNX', 'FVX',
                'GF', 'COF', 'GBPUSD', 'FTSE100', 'total_rep_pf',
                'volatility_ftse', 'volatility_pf', 'spread_ma5', 'spread_ma20', 'rsi_spread']

    target = ['spread']

    train_size = int(len(merged_df) * 0.8)
    train_data = merged_df.iloc[:train_size]
    test_data = merged_df.iloc[train_size:]

    # feature_scaler = MinMaxScaler(feature_range=(0, 1))
    # target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(train_data[features])
    target_scaler.fit(train_data[target])

    train_feature_scaled = feature_scaler.transform(train_data[features])
    train_target_scaled = target_scaler.transform(train_data[target])

    test_feature_scaled = feature_scaler.transform(test_data[features])
    test_target_scaled = target_scaler.transform(test_data[target])

    X_train, y_train = create_sequences(train_feature_scaled, train_target_scaled, sequence_length)
    X_test, y_test = create_sequences(test_feature_scaled, test_target_scaled, sequence_length)


    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, train_data,test_data,merged_df,features



# In[137]:


# Function to calculate RSI
def calculate_rsi(series, window=14):
    """Calculate RSI for a price series"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# In[138]:


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    return np.array(X_seq), np.array(y_seq)


# In[139]:


def build_lstm_model(X_train, y_train, X_test, y_test):

    n_features = X_train.shape[2]

    lr=3.28920104987804e-05
    model = Sequential()
    model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


# In[140]:


economic_indicators=log_return_eco

X_train, X_test, y_train, y_test, feature_scaler,target_scaler,train_data,test_data,merged_df,features= prepare_data_for_lstm(
   spread_df , economic_indicators, 20
)

model, history = build_lstm_model(X_train, y_train, X_test, y_test)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# In[141]:


sequence_length=20
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred = target_scaler.inverse_transform(train_pred)
test_pred = target_scaler.inverse_transform(test_pred)

train_indices = train_data.index[sequence_length:]
test_indices = test_data.index[sequence_length:]

train_predictions_df = pd.DataFrame(
    data=train_pred,
    columns=['predicted_spread'],
    index=train_indices[:len(train_pred)]  # Ensure matching lengths
)
train_predictions_df['actual_spread'] = train_data.loc[train_indices[:len(train_pred)], 'spread'].values
train_predictions_df['dataset'] = 'train'

test_predictions_df = pd.DataFrame(
    data=test_pred,
    columns=['predicted_spread'],
    index=test_indices[:len(test_pred)]  # Ensure matching lengths
)
test_predictions_df['actual_spread'] = test_data.loc[test_indices[:len(test_pred)], 'spread'].values
test_predictions_df['dataset'] = 'test'

# Combine predictions for full view
all_predictions_df = pd.concat([train_predictions_df, test_predictions_df])
all_predictions_df


# In[ ]:





# In[142]:


z_score_window=20
test_predictions = all_predictions_df[all_predictions_df['dataset'] == 'test'].copy()

actual_spread = test_predictions['actual_spread']
predicted_spread = test_predictions['predicted_spread']

error = actual_spread - predicted_spread

rolling_mean = error.rolling(window=z_score_window).mean()
rolling_std = error.rolling(window=z_score_window).std()
z_score = (error - rolling_mean) / rolling_std

signals = pd.DataFrame(index=test_predictions.index)
signals['actual_spread'] = actual_spread
signals['predicted_spread'] = predicted_spread
signals['error'] = error
signals['z_score'] = z_score
signals['signal'] = 0

signals.loc[z_score <= -1.5, 'signal'] = 1    # Buy when actual is lower than predicted
signals.loc[z_score >= 1.5, 'signal'] = -1    # Sell when actual is higher than predicted


# In[ ]:





# In[ ]:





# In[143]:


signals_df=signals
signals_df


# In[144]:


window=5
eval_df = signals.copy()
eval_df['signal'] = eval_df['signal'].fillna(0)
eval_df['forward_spread_change'] = eval_df['actual_spread'].shift(-window) - eval_df['actual_spread']

buy_success = ((eval_df['signal'] == 1) & (eval_df['forward_spread_change'] > 0))
sell_success = ((eval_df['signal'] == -1) & (eval_df['forward_spread_change'] < 0))
hold_success = ((eval_df['signal'] == 0) &
               (abs(eval_df['forward_spread_change']) < eval_df['forward_spread_change'].std() * 0.5))

buy_success_rate = buy_success.sum() / (eval_df['signal'] == 1).sum() if (eval_df['signal'] == 1).sum() > 0 else 0
sell_success_rate = sell_success.sum() / (eval_df['signal'] == -1).sum() if (eval_df['signal'] == -1).sum() > 0 else 0
hold_success_rate = hold_success.sum() / (eval_df['signal'] == 0).sum() if (eval_df['signal'] == 0).sum() > 0 else 0

overall_success = buy_success.sum() + sell_success.sum() + hold_success.sum()
total_signals = len(eval_df.dropna())
overall_success_rate = overall_success / total_signals if total_signals > 0 else 0

evaluation_metrics= {
    'buy_success_rate': buy_success_rate,
    'sell_success_rate': sell_success_rate,
    'hold_success_rate': hold_success_rate,
    'overall_success_rate': overall_success_rate,
    'buy_count': (eval_df['signal'] == 1).sum(),
    'sell_count': (eval_df['signal'] == -1).sum(),
    'hold_count': (eval_df['signal'] == 0).sum()
}


# In[ ]:






# In[145]:


print("\nStrategy Evaluation:")
print(f"Buy Signal Success Rate: {evaluation_metrics['buy_success_rate']:.2%}")
print(f"Sell Signal Success Rate: {evaluation_metrics['sell_success_rate']:.2%}")
print(f"Hold Signal Success Rate: {evaluation_metrics['hold_success_rate']:.2%}")
print(f"Overall Success Rate: {evaluation_metrics['overall_success_rate']:.2%}")
print(f"Buy Signals: {evaluation_metrics['buy_count']}")
print(f"Sell Signals: {evaluation_metrics['sell_count']}")
print(f"Hold Signals: {evaluation_metrics['hold_count']}")



# In[ ]:





# In[146]:


plt.figure(figsize=(14, 20))

plt.subplot(3, 1, 1)

# Plot training data predictions
train_data = all_predictions_df[all_predictions_df['dataset'] == 'train']
plt.plot(train_data.index, train_data['actual_spread'],
         color='blue', label='Actual (Train)')
plt.plot(train_data.index, train_data['predicted_spread'],
         color='green', linestyle='--', label='Predicted (Train)')

# Plot test data predictions
test_data = all_predictions_df[all_predictions_df['dataset'] == 'test']
plt.plot(test_data.index, test_data['actual_spread'],
         color='red', label='Actual (Test)')
plt.plot(test_data.index, test_data['predicted_spread'],
         color='orange', linestyle='--', label='Predicted (Test)')

plt.title('Actual vs Predicted Spread')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)

# Plot 3: Trading signals
plt.subplot(3, 1, 2)
plt.plot(signals_df['actual_spread'], label='Actual Spread', color='blue')

# Plot buy signals
buy_signals = signals_df[signals_df['signal'] == 1].copy()
if not buy_signals.empty:
    plt.scatter(buy_signals.index, buy_signals['actual_spread'],
            color='green', marker='^', s=100, label='Buy')

# Plot sell signals
sell_signals = signals_df[signals_df['signal'] == -1].copy()
if not sell_signals.empty:
    plt.scatter(sell_signals.index, sell_signals['actual_spread'],
            color='red', marker='v', s=100, label='Sell')

plt.title('Trading Signals')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[147]:


#Feature importance
baseline_loss = model.evaluate(X_test, y_test, verbose=0)
importance_scores = []

for i in range(X_test.shape[2]):
    X_test_permuted = X_test.copy()
    X_test_permuted[:, :, i] = np.random.permutation(X_test_permuted[:, :, i])

    permuted_loss = model.evaluate(X_test_permuted, y_test, verbose=0)
    importance = permuted_loss - baseline_loss
    importance_scores.append(importance)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance_scores
})

importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df)


# In[ ]:





# In[148]:


fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[2, 0])
ax1.hist(signals_df['z_score'].dropna(), bins=30, color='skyblue', edgecolor='black')
ax1.axvline(x=1.5, color='red', linestyle='--', label='Sell Threshold')
ax1.axvline(x=-1.5, color='green', linestyle='--', label='Buy Threshold')
ax1.set_title('Z-Score Distribution')
ax1.legend()
ax1.grid(True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[149]:


# indicator = 'VIX'
# plt.scatter(merged_df[indicator], merged_df['spread'], alpha=0.5)
# plt.title(f'Spread vs {indicator}')
# plt.xlabel(indicator)
# plt.ylabel('Spread')
# plt.grid(True)


# In[150]:


get_ipython().system('pip install shap')


# In[151]:


# import shap

# masker = shap.maskers.Independent(X_test)
# explainer = shap.Explainer(model, masker)
# shap_values = explainer(X_test,max_evals='auto')

# shap.summary_plot(shap_values, features=features)


# In[ ]:





# In[ ]:





# In[ ]:





# In[152]:


# !pip install keras_tuner

def get_tuned_lstm(hp):  # Accept hp as an argument
    n_features = X_train.shape[2]

    model = Sequential()

    # Use hyperparameters for tuning the LSTM layers
    model.add(LSTM(
        hp.Int('units_1', min_value=10, max_value=150, step=20),
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], n_features)
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(
        hp.Int('units_2', min_value=10, max_value=100, step=20),
        activation='relu'
    ))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    # Use hyperparameter for learning rate tuning
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log')),
        loss='mse'
    )

    return model


# In[153]:


from keras_tuner import RandomSearch

tuner = RandomSearch(
    get_tuned_lstm,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='lstm_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))



# In[154]:


best_trial = tuner.oracle.get_best_trials()[0]
best_hyperparameters = best_trial.hyperparameters.values
best_hyperparameters


# In[ ]:





# In[ ]:


import joblib

model.save("hackathon_attendance_model.h5")
joblib.dump(scaler, "scaler.pkl")


# In[ ]:
