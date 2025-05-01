import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()  # Automatically loads variables from the .env file

# Read private key from environment variables (set in Railway)
raw_private_key = os.getenv("private_key")

if raw_private_key:
    # Replace the literal "\n" with actual newlines
    private_key = raw_private_key.replace("\\n", "\n")
else:
    raise ValueError("Missing private key from environment")

credentials_info = {
    "type": "service_account",
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    "private_key": private_key,
    "client_email": os.getenv("client_email"),
    "client_id": os.getenv("client_id"),
    "auth_uri": os.getenv("auth_uri"),
    "token_uri": os.getenv("token_uri"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_cert_url"),
    "client_x509_cert_url": os.getenv("client_cert_url"),
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
