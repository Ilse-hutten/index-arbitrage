from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def rolling_pca_weights(X_log, n_stocks, window_pca, n_pcs):
    """
    Compute PCA-based portfolio weights for a specific date.

    Parameters:
    - X_log: DataFrame of log returns
    - n_stocks: Number of top stocks to select each window
    - window_pca: Number of days the pca is calculated over
    - n_pcs: Number of principal components to sum

    Returns:
    - rep_pf: DataFrame with stocks as columns and weights as the values
    """

    # Initialize
    dates = X_log.index[window_pca:]
    summed_pcs_full = {}

    # Rolling PCA computation
    def compute_rolling_pca(window_start):
        pca_roll = PCA()
        window_data = X_log.iloc[window_start:window_start + window_pca, :]  # Full rolling window
        pca_roll.fit(window_data)
        loadings_matrix = pca_roll.components_.T
        summed_values = loadings_matrix[:, :n_pcs].sum(axis=1)
        summed_pcs_full[dates[window_start]] = pd.Series(summed_values, index=X_log.columns)

    # Run rolling PCA for all windows
    for start in range(len(X_log) - window_pca):
       compute_rolling_pca(start)

    # Combine all summed PCs into DataFrame
    summed_pcs_full_df = pd.DataFrame(summed_pcs_full).T
    summed_pcs_full_df.index.name = "date"

    # Initialize daily weights list
    daily_weights = []

    # Calculate weights for each date
    for date in summed_pcs_full_df.index:
        # Get summed PCs for all stocks for date
        daily_values = summed_pcs_full_df.loc[date]

        # Select the top X stocks for this date
        top_stocks = daily_values.nlargest(n_stocks)

        # Calculate stock weights
        portfolio_weights = top_stocks / top_stocks.sum()

        # Create a row of weights with 0 for stocks not in the top X
        row_weights = pd.Series(0.0, index=summed_pcs_full_df.columns)  # Initialize with zeros
        row_weights[top_stocks.index] = portfolio_weights  # Update weights for the top stocks

        # Add the row of weights to the daily weights list
        daily_weights.append(row_weights)

    # Combine daily weights into a DataFrame
    daily_weights_df = pd.DataFrame(daily_weights, index=summed_pcs_full_df.index)

    return daily_weights_df
