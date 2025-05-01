import os
from google.cloud import bigquery
from google.oauth2 import service_account

# Build credentials from environment variables
credentials_info = {
    "type": "service_account",
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_CERT_URL"),
}
credentials = service_account.Credentials.from_service_account_info(credentials_info)

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
