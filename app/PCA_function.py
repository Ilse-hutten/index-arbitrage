#!/usr/bin/env python
# coding: utf-8


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


# In[35]:


# Define input variables
n_stocks = 30
window = 30 # period the trading strat goes
n_pcs = 3
pca_date = '2023-06-16'
z_window=60 # to calc z score

# Get weights
rep_pf = rolling_pca_weights(X_log, n_stocks, window, n_pcs, pca_date)


# In[36]:


rep_pf
