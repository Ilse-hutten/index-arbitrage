from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocessing_X(stocks_df):
    stocks_df.set_index('date', inplace=True)

    # Create log returns to remove stationarity
    log_returns = np.log(stocks_df / stocks_df.shift(1))

    # Drop NaN values caused by the shift
    log_returns = log_returns.dropna()
    X_log = log_returns.copy()
    stock_log_features = X_log.columns

    # Scaling data
    scaler = StandardScaler()
    scaler.fit(X_log)
    X_log = pd.DataFrame(scaler.transform(X_log), columns=stock_log_features, index=log_returns.index)
    return X_log

df= preprocessing_X()
print(df)
