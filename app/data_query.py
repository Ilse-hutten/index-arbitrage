import os
from google.cloud import bigquery

# Path to your BigQuery JSON key file
#json_key_path = "big_query_key.json"

# Set environment variable for authentication
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path



def fetch_data(dataset: str, table: str):

    query = f"SELECT * FROM `lewagon-statistical-arbitrage.{dataset}.{table}` ORDER BY date"

    client = bigquery.Client()  # Initialize BigQuery client
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

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame
