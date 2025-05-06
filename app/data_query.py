import os
from google.cloud import bigquery
from google.oauth2 import service_account
from app.app import credentials
from google.cloud import bigquery

client = bigquery.Client(credentials=credentials)

def fetch_data(dataset: str, table: str):
    query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"

    # Pass the explicit credentials to BigQuery client
    client = bigquery.Client(credentials=credentials)
    return client.query(query).to_dataframe()  # Run query and return DataFrame


# Fetching functions for specific datasets
def fetch_NASDAQ100_index():
    return fetch_data("NASDAQ100", "NASDAQ100_index")

def fetch_NASDAQ100_all_components():
    return fetch_data("NASDAQ100", "NASDAQ100_all_components")

def fetch_SP500_index():
    return fetch_data("SP500", "SP500_index")

def fetch_SP500_all_components():
    return fetch_data("SP500", "SP500_all_components")

def fetch_ftse100_index():
    return fetch_data("FTSE100", "FTSE100_index")

def fetch_ftse100_all_components():
    return fetch_data("FTSE100", "FTSE100_all_components")

def eco_df():
    return fetch_data("ECO_DF","economic_combination")

def fetch_CRYPTO_index():
    return fetch_data("CRYPTO", "CRYPTO_INDEX")

def fetch_CRYPTO_all_components():
    return fetch_data("CRYPTO", "CRYPTO_ALL_COMPONENTS")

# print(fetch_ftse100_all_components())
