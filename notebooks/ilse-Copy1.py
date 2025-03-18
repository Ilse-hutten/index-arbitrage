#!/usr/bin/env python
# coding: utf-8

# # PCA for PF Construction

# In[ ]:





# In[1]:


from sklearn.decomposition import PCA
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import sys
import os
sys.path.append(os.path.abspath("../app"))
from frame import Frame
data = Frame()
df = data.dataset()


# In[3]:


df.set_index('date', inplace=True)


# In[4]:


df


# In[ ]:





# In[5]:


# Create log returns to remove stationarity
log_returns = np.log(df / df.shift(1))

# Drop NaN values caused by the shift
log_returns = log_returns.dropna().drop(columns="FTSE100")


# In[6]:


# Creating X of the closing prices (absolute prices)
X_abs = df.drop(columns="FTSE100").dropna()
stock_features = X_abs.columns
X_abs.shape


# In[7]:


# Creating X of the closing prices (log returns)
X_log = log_returns.copy()
stock_log_features = X_log.columns
X_log.shape


# In[8]:


# Preprocessing ABS (data must be centered around its mean before PCA)
scaler = StandardScaler()
scaler.fit(X_abs)
X_abs = pd.DataFrame(scaler.transform(X_abs), columns=stock_features)
X_abs


# In[10]:


# Preprocessing LOG (data must be centered around its mean before PCA)
scaler.fit(X_log)
X_log = pd.DataFrame(scaler.transform(X_log), columns=stock_log_features, index=log_returns.index)
X_log


# In[12]:


# Compute Principal Components
pca = PCA()
pca.fit(X_abs)


# 

# In[13]:


# Access PCs
W = pca.components_
# Print PCs as COLUMNS
T = pd.DataFrame(W.T,
                 index=stock_features,
                 columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])
T


# In[14]:


# Compute Principal Components log returns
pca2 = PCA()
pca2.fit(X_log)


# In[15]:


# Access PCs
W2 = pca2.components_
# Print PCs as COLUMNS
T_log = pd.DataFrame(W2.T,
                 index=stock_log_features,
                 columns=[f'PC{i}' for i in range(1, pca2.n_components_+1)])
T_log


# In[16]:


# pca.explained_variance_ratio_ and pca2.explained_variance_ratio_ (absolute vs log return)
explained_variance_1 = pca.explained_variance_ratio_
explained_variance_2 = pca2.explained_variance_ratio_

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot the first PCA - abs
axes[0].plot(explained_variance_1)
axes[0].set_title('PCA 1 - Explained Variance using abs prices')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('% Explained Variance')

# Plot the second PCA - log
axes[1].plot(explained_variance_2)
axes[1].set_title('PCA 2 - Explained Variance using log return')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('% Explained Variance')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[17]:


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# first PCA - ABS
axes[0].plot(np.cumsum(explained_variance_1))
axes[0].set_title('cumulated share of explained variance using stock prices')
axes[0].set_xlabel('# of principal component used')

# second PCA - log
axes[1].plot(np.cumsum(explained_variance_2))
axes[1].set_title('cumulated share of explained variance using log return')
axes[1].set_xlabel('# of principal component used')

# Adjust the y-axis scale
axes[0].set_ylim(0.4, 1.0)  # Adjust scale for subplot 1
axes[1].set_ylim(0.2, 1.0)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[18]:


# Picking stocks most closely mimicking the index (based on log returns method)

# Keeping only PC1 and PC2
T_log = T_log.iloc[:, :4]

# adding column showing cum PC1+PC2
T_log["PC_sum"] = T_log["PC1"]+T_log["PC2"]+T_log["PC3"]+T_log["PC4"]

# Sort by PC_sum
T_log_sorted = T_log.sort_values("PC_sum", ascending=False)
T_log_sorted = T_log_sorted.reset_index()
T_log_sorted.rename(columns={T_log_sorted.columns[0]: "Stocks" }, inplace = True)

# Calculate the sum of 'PC_sum' for the top 5 rows
top_5_sum = T_log_sorted["PC_sum"].head(5).sum()

# Add the 'pf_weights' column by dividing each stock's 'PC_sum' by the top 5 sum
T_log_sorted["pf_weights"] = T_log_sorted["PC_sum"] /top_5_sum
# Set weights to 0 for rows beyond the top 5
T_log_sorted.loc[5:, "pf_weights"] = 0

T_log_sorted.head(10)


# In[19]:


import matplotlib.dates as mdates
# Index the data by dividing each stock's prices by its first value
indexed_data = df / df.iloc[0] * 100

# Retrieve the top 5 stocks from the "Stocks" column of T_log_sorted
top_5_stocks = T_log_sorted["Stocks"].head(5)

# Plot the FTSE index
plt.plot(indexed_data["FTSE100"], label="FTSE100", linewidth=2)

# Plot each of the top 5 stocks dynamically
for stock in top_5_stocks:
    if stock in indexed_data.columns:  # Ensure the stock is in the data
        plt.plot(indexed_data[stock], label=stock)

# Add a legend to distinguish the lines
plt.legend()

# Format the dates on the x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show dates every 1 month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as "Month Year"

# Add titles and labels
plt.title("Indexed Closing Prices of Top 5 Stocks and FTSE")
plt.xlabel("Time")
plt.ylabel("Indexed Price (Base = 100)")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()


# In[20]:


# Extract the top 5 stocks and their weights to make a replicating portfolio
top_5_stocks = T_log_sorted["Stocks"].head(5)
top_5_weights = T_log_sorted["pf_weights"].head(5)

# Calculate the weighted sumproduct (dot product of weights and indexed returns)
weighted_returns = (indexed_data[top_5_stocks] * top_5_weights.values).sum(axis=1)

# Plot the weighted returns
plt.plot(weighted_returns, label="Synthetic Portfolio", color="blue", linewidth=2)

# Plot the FTSE index for comparison
plt.plot(indexed_data["FTSE100"], label="FTSE", color="orange", linestyle="--")

# Add legend, labels, and title
plt.legend()
plt.title("Synthetic Portfolio vs FTSE Index")
plt.xlabel("Time")
plt.ylabel("Weighted Indexed Returns")
plt.xticks(rotation=45)

# Show the plot


# Rolling PCA

# In[55]:


n_stocks = 10
window = 100 # n of trading days
dates = X_log.index[window:]  # Align dates with the end of each rolling window

# Placeholder for summed PCs for all stocks (full data)
summed_pcs_full = {}

def rolling_pca(window_start):
    pca_roll = PCA()
    # Create the rolling window data excluding today's data
    window_data = X_log.iloc[window_start:window_start + window - 1, :]  # Exclude today's row
    # Fit PCA to the rolling window
    pca_roll.fit(window_data)
    # Extract loadings (components matrix)
    loadings_matrix = pca_roll.components_.T  # Transpose to get stocks as rows

    # Sum the first 4 PCs for each stock
    summed_values = loadings_matrix[:, :4].sum(axis=1)  # Sum across the first 4 PCs
    summed_pcs_full[dates[window_start]] = pd.Series(summed_values, index=X_log.columns)  # Store as Series

# Iterate through rolling windows
for start in range(len(X_log) - window):
    rolling_pca(start)

# Combine results into a full DataFrame (dates as rows, stocks as columns)
summed_pcs_full_df = pd.DataFrame(summed_pcs_full).T  # Transpose to align dates as rows
summed_pcs_full_df.index.name = "Date"  # Set index name


# In[56]:


# Placeholder for daily portfolio weights
daily_portfolio = []

# Loop through each date
for date in summed_pcs_full_df.index:
    # Get summed PCs for all stocks on this date
    daily_values = summed_pcs_full_df.loc[date]

    # Select the top X stocks for this date
    top_stocks = daily_values.nlargest(n_stocks)  # Top 10 stocks by summed PCs

    # Normalize the summed PCs to use as portfolio weights
    portfolio_weights = top_stocks / top_stocks.sum()  # calc weights

    # Store the portfolio details (date, stocks, weights)
    portfolio_details = {
        "Date": date,
        "Stocks": list(top_stocks.index),
        "Weights": list(portfolio_weights.values)
    }
    daily_portfolio.append(portfolio_details)

# Convert to a structured DataFrame
daily_portfolio_df = pd.DataFrame(daily_portfolio)


# In[57]:


# Placeholder for daily weights across all stocks
daily_weights = []

# Loop through each date in the summed PCs DataFrame
for date in summed_pcs_full_df.index:
    # Get summed PCs for all stocks on this date
    daily_values = summed_pcs_full_df.loc[date]

    # Select the top X stocks for this date
    top_stocks = daily_values.nlargest(n_stocks)  # Top X stocks by summed PCs

    # Calculate stock weights
    portfolio_weights = top_stocks / top_stocks.sum()

    # Create a row of weights with 0 for stocks not in the top X
    row_weights = pd.Series(0, index=summed_pcs_full_df.columns)  # Initialize with zeros
    row_weights[top_stocks.index] = portfolio_weights  # Update weights for the top stocks

    # Add the row of weights to the daily weights list
    daily_weights.append(row_weights)

# Create the final DataFrame of daily weights
daily_weights_df = pd.DataFrame(daily_weights, index=summed_pcs_full_df.index)


# In[66]:


daily_weights_df.to_csv("daily_weights.csv")


# In[58]:


daily_weights_df.head(20)


# Example trading results using one specific date for pca (not rolling)

# In[59]:


pca_date = '2022-10-14'

# Filter stocks with values > 0 on the specific date
filtered_columns = daily_weights_df.loc[pca_date][daily_weights_df.loc[pca_date] > 0].index
rep_pf = daily_weights_df.loc[[pca_date], filtered_columns]


# In[60]:


rep_pf


# In[61]:


rep_pf_log_returns_daily = log_returns[rep_pf.columns]
rep_pf_log_returns_daily


# In[29]:


# weights of the rep_pf times the daily log returns, summed together in the total_rep_pf column
rep_pf_results = rep_pf_log_returns_daily.mul(rep_pf.iloc[0], axis=1)
rep_pf_results["total_rep_pf"] = rep_pf_results.sum(axis=1)
rep_pf_results


# In[30]:


# Calculate log returns FTSE100
FTSE_log_return = np.log(df["FTSE100"] / df["FTSE100"].shift(1)).dropna()
FTSE_log_return


# In[31]:


spread_df = pd.DataFrame()  # Create a new DataFrame for the spread
spread_df["spread"] = FTSE_log_return - rep_pf_results["total_rep_pf"]  # Add a column named "spread"
spread_df


# In[35]:


from scipy.stats import zscore

end_date = pd.to_datetime(pca_date) + pd.Timedelta(days=30)

# Filter the DataFrame to include only the target date range
target_df = spread_df.loc[pca_date:end_date]

# Function to calculate rolling z-scores using scipy.stats.zscore
def rolling_z_score(series):
    if len(series) == 0 or series.std() == 0:  # Handle cases with no variance
        return 0
    return zscore(series)[-1]  # Get the z-score of the last element in the rolling window

# Calculate rolling z-score (exclude today's value using closed='left')
spread_df['z_score'] = (
    spread_df['spread']
    .rolling(window=60, min_periods=60, closed='both')  # Rolling window excluding the current day "closed = left", incl today is closed = both
    .apply(lambda x: rolling_z_score(x), raw=False)
)

# Apply the calculation only for the specified date range
rolling_z_score = spread_df.loc[pca_date:end_date]


# In[36]:


rolling_z_score['trade_signal'] = 0
current_position = 0  # 1 for long, -1 for short, 0 for no position

for i in range(len(rolling_z_score)):
    z = rolling_z_score.iloc[i]['z_score']

    if current_position == 0:
        if z < -2:
            current_position = 1  # Long FTSE
        elif z > 2:
            current_position = -1  # Short FTSE
    elif current_position == 1:
        if z > -0.5:
            current_position = 0
    elif current_position == -1:
        if z < 0.5:
            current_position = 0

    rolling_z_score.iloc[i, rolling_z_score.columns.get_loc('trade_signal')] = current_position


# In[68]:


# Shift trade signal to apply NEXT day's return
rolling_z_score['shifted_signal'] = rolling_z_score['trade_signal'].shift(1)

# Calculate strategy return based on shifted signal
rolling_z_score['strategy_return'] = np.where(
    rolling_z_score['shifted_signal'] == 1,  # Long FTSE, short replication portfolio
    FTSE_log_return.loc[pca_date:end_date].values - rep_pf_results["total_rep_pf"].loc[pca_date:end_date].values,
    np.where(
        rolling_z_score['shifted_signal'] == -1,  # Short FTSE, long replication portfolio
        rep_pf_results["total_rep_pf"].loc[pca_date:end_date].values - FTSE_log_return.loc[pca_date:end_date].values,
        FTSE_log_return.loc[pca_date:end_date].values  # Hold FTSE when signal is 0
    )
)

# Align index
rolling_z_score = rolling_z_score.loc[pca_date:end_date]

# Calculate cumulative return
rolling_z_score['cumulative_return'] = (1 + rolling_z_score['strategy_return']).cumprod()

rolling_z_score.tail(50)


# In[67]:


rolling_z_score['cumulative_ftse_return'] = (1 + FTSE_log_return.loc[pca_date:end_date]).cumprod()

# Plot the cumulative strategy return and cumulative FTSE return
plt.figure(figsize=(12, 6))
plt.plot(rolling_z_score.index, rolling_z_score['cumulative_return'], label='Cumulative Strategy Return', color='blue')
plt.plot(rolling_z_score.index, rolling_z_score['cumulative_ftse_return'], label='Cumulative FTSE Return', color='orange')
plt.title('Cumulative Strategy Return vs Cumulative FTSE Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.show()


# # Creating Functions of Code

# In[ ]:


def rolling_pca_weights(X_log, n_stocks, window, n_pcs, pca_date):
    """
    Compute PCA-based portfolio weights for a specific date.

    Parameters:
    - X_log: DataFrame of log returns
    - n_stocks: Number of top stocks to select each window
    - window: Number of trading days per rolling window
    - n_pcs: Number of principal components to sum
    - pca_date: Date (as string or Timestamp) to compute weights for

    Returns:
    - rep_pf: DataFrame with stocks as columns and weights as the values
    """

    # Initialize
    dates = X_log.index[window:]
    summed_pcs_full = {}

    # Rolling PCA computation
    def compute_rolling_pca(window_start):
        pca_roll = PCA()
        window_data = X_log.iloc[window_start:window_start + window - 1, :]
        pca_roll.fit(window_data)
        loadings_matrix = pca_roll.components_.T
        summed_values = loadings_matrix[:, :n_pcs].sum(axis=1)
        summed_pcs_full[dates[window_start]] = pd.Series(summed_values, index=X_log.columns)

    # Run rolling PCA
    for start in range(len(X_log) - window):
        compute_rolling_pca(start)

    # Combine all summed PCs into DataFrame
    summed_pcs_full_df = pd.DataFrame(summed_pcs_full).T
    summed_pcs_full_df.index.name = "Date"

    # Check if requested date is available
    if pd.to_datetime(pca_date) not in summed_pcs_full_df.index:
        raise ValueError(f"The date {pca_date} is not available in the data.")

    # Calculate weights for the specified date
    daily_values = summed_pcs_full_df.loc[pd.to_datetime(pca_date)]
    top_stocks = daily_values.nlargest(n_stocks)
    portfolio_weights = top_stocks / top_stocks.sum()

    # Create a one-row DataFrame with stocks as columns and weights as the values
    rep_pf = pd.DataFrame([portfolio_weights.values], columns=portfolio_weights.index)

    return rep_pf


# In[ ]:


rep_pf


# In[ ]:


def strategy_returns(rep_pf, log_returns, FTSE_prices, pca_date, window, z_window):
    """
    Computes strategy returns based on PCA-replicated portfolio and z-score based trading signals.
    
    Parameters:
    - rep_pf: DataFrame with portfolio weights on pca_date
    - log_returns: Full DataFrame of log returns
    - FTSE_prices: Series of FTSE prices
    - pca_date: Date for PCA weights (string or datetime)
    - end_days: How many days after pca_date to calculate returns
    - z_window: Rolling window for z-score
    
    Returns:
    - Tuple of DataFrame with strategy performance and a metrics comparison DataFrame
    """
    
    # 1. Calculate daily log returns of replication portfolio
    rep_pf_log_returns_daily = log_returns[rep_pf.columns]
    rep_pf_results = rep_pf_log_returns_daily.mul(rep_pf.iloc[0], axis=1)
    rep_pf_results["total_rep_pf"] = rep_pf_results.sum(axis=1)
    
    # 2. Calculate FTSE log returns
    FTSE_log_return = np.log(FTSE_prices / FTSE_prices.shift(1)).dropna()
    
    # 3. Compute spread
    spread_df = pd.DataFrame()
    spread_df["spread"] = FTSE_log_return - rep_pf_results["total_rep_pf"]
    
    # 4. Rolling z-score calculation
    spread_df['z_score'] = (
        spread_df['spread']
        .rolling(window=z_window, min_periods=z_window, closed='both')
        .apply(lambda x: zscore(x)[-1] if len(x) > 1 else 0, raw=False)
    )
    
    # 5. Filter relevant date range
    end_date = pd.to_datetime(pca_date) + pd.Timedelta(days=window)
    rolling_z_score = spread_df.loc[pca_date:end_date]
    
    # 6. Trading logic
    rolling_z_score['trade_signal'] = 0
    current_position = 0
    
    for i in range(len(rolling_z_score)):
        z = rolling_z_score.iloc[i]['z_score']

        if current_position == 0:
            if z < -2:
                current_position = 1  # Long FTSE
            elif z > 2:
                current_position = -1  # Short FTSE
        elif current_position == 1:
            if z > -0.5:
                current_position = 0
        elif current_position == -1:
            if z < 0.5:
                current_position = 0

        rolling_z_score.iloc[i, rolling_z_score.columns.get_loc('trade_signal')] = current_position
    
    # 7. Shift signal to apply on next day
    rolling_z_score['shifted_signal'] = rolling_z_score['trade_signal'].shift(1)
    
    # 8. Calculate strategy return
    ftse_returns = FTSE_log_return.loc[pca_date:end_date]
    rep_pf_returns = rep_pf_results["total_rep_pf"].loc[pca_date:end_date]
    
    rolling_z_score['strategy_return'] = np.where(
        rolling_z_score['shifted_signal'] == 1,
        ftse_returns.values - rep_pf_returns.values,
        np.where(
            rolling_z_score['shifted_signal'] == -1,
            rep_pf_returns.values - ftse_returns.values,
            ftse_returns.values  # Hold FTSE
        )
    )
    
    # 9. Calculate cumulative returns
    rolling_z_score['cumulative_return'] = (1 + rolling_z_score['strategy_return']).cumprod()
    rolling_z_score['cumulative_ftse_return'] = (1 + ftse_returns).cumprod()
    
    # 10. Compute Performance Metrics
    total_strategy_return = rolling_z_score['cumulative_return'].iloc[-1] - 1
    total_ftse_return = rolling_z_score['cumulative_ftse_return'].iloc[-1] - 1
    
    annualized_strategy_return = ((1 + total_strategy_return) ** (252 / len(rolling_z_score))) - 1
    annualized_ftse_return = ((1 + total_ftse_return) ** (252 / len(rolling_z_score))) - 1

    strategy_volatility = rolling_z_score['strategy_return'].std() * np.sqrt(252)
    ftse_volatility = ftse_returns.std() * np.sqrt(252)

    # Sharpe ratio assumes a risk-free rate of 0 for simplicity
    strategy_sharpe = annualized_strategy_return / strategy_volatility
    ftse_sharpe = annualized_ftse_return / ftse_volatility
    
    # Create Metrics DataFrame with the desired format
    metrics_df = pd.DataFrame({
        "Replication Strategy": [
            total_strategy_return * 100,
            annualized_strategy_return * 100,
            strategy_volatility * 100,
            strategy_sharpe
        ],
        "FTSE": [
            total_ftse_return * 100,
            annualized_ftse_return * 100,
            ftse_volatility * 100,
            ftse_sharpe
        ]
    }, index=[
        "Total Return (%)",
        "Annualized Return (%)",
        "Volatility (%)",
        "Sharpe Ratio"
    ])

    # 11. Plot Cumulative Returns
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_z_score.index, rolling_z_score['cumulative_return'], label='Strategy Return', color='blue')
    plt.plot(rolling_z_score.index, rolling_z_score['cumulative_ftse_return'], label='FTSE Return', color='orange')
    plt.title('Cumulative Strategy Return vs FTSE Return')
    plt.legend()
    plt.grid()
    plt.show()
    
    return rolling_z_score, metrics_df


# In[ ]:


# Define input variables
n_stocks = 10
window = 30 # period the trading strat goes
n_pcs = 4
pca_date = '2022-10-14'
z_window=60 # to calc z score

# Get weights
rep_pf = rolling_pca_weights(X_log, n_stocks, window, n_pcs, pca_date)


# In[ ]:


rolling_z_score, metrics_df = strategy_returns(rep_pf, log_returns, df["FTSE100"], pca_date, window, z_window)
metrics_df


# Check for optimal parameters

# In[ ]:


import itertools

# Define ranges for input variables
n_stocks_range = range(40,51, 10)  # From 10 to 60 in steps of 10
window_range = range(40, 61, 20)   # From 40 to 100 in steps of 10
n_pcs_range = range(1, 4,1)          # From 1 to 20
benchmark_dates = ['2023-06-01', '2023-12-01', '2024-03-01']  # Dates to evaluate

# Initialize a list to store results
global_results = []

# Iterate through all combinations of n_stocks, window, and n_pcs
for n_stocks, window, n_pcs in itertools.product(n_stocks_range, window_range, n_pcs_range):
    try:
        # Initialize performance metrics across all benchmark dates
        date_results = []
        
        for pca_date in benchmark_dates:
            try:
                # Step 1: Generate portfolio weights
                rep_pf = rolling_pca_weights(X_log, n_stocks, window, n_pcs, pca_date)
                
                # Step 2: Evaluate strategy returns
                rolling_z_score, metrics_df = strategy_returns(rep_pf, log_returns, df["FTSE100"], pca_date, window, z_window)
                
                # Step 3: Extract total returns for this date
                total_strategy_return = metrics_df.loc["Total Return (%)", "Replication Strategy"]
                total_ftse_return = metrics_df.loc["Total Return (%)", "FTSE"]
                return_diff = total_strategy_return - total_ftse_return
                
                # Append the result for this date
                date_results.append(return_diff)
            
            except Exception as e:
                # Catch errors for this specific date
                print(f"Error for date {pca_date} with n_stocks={n_stocks}, window={window}, n_pcs={n_pcs}: {e}")
        
        # Aggregate results across all dates
        if date_results:
            avg_return_diff = sum(date_results) / len(date_results)  # Mean difference
            global_results.append({
                "n_stocks": n_stocks,
                "window": window,
                "n_pcs": n_pcs,
                "avg_return_diff (%)": avg_return_diff
            })

    except Exception as e:
        # Catch errors for parameter combination
        print(f"Error for combination n_stocks={n_stocks}, window={window}, n_pcs={n_pcs}: {e}")

# Convert global results to a DataFrame
global_results_df = pd.DataFrame(global_results)

# Identify the best combination
best_global_result = global_results_df.loc[global_results_df['avg_return_diff (%)'].idxmax()]

# Display the best parameters
print("Best Combination Across All Dates:")
print(best_global_result)

# Display all results for visualization
print("\nAll Parameter Combinations and Their Average Performance:")
print(global_results_df)


# In[ ]:


global_results_df


# # Predicting Spread with LSTM / NN

# In[ ]:


import yfinance as yf

# 10-year Treasury yield (^TNX)
X_df = yf.download("^TNX", start="2022-06-29", end="2025-03-10")
print(X_df)


# In[ ]:


X_df

