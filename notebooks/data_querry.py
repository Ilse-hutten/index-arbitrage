import os
from google.cloud import bigquery

# Path to your BigQuery JSON key file
#json_key_path = "big_query_key.json"

# Set environment variable for authentication
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path


def fetch_NASDAQ100_index():
    """Fetch NASDAQ100 index data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.NASDAQ100.NASDAQ100_index` ORDER BY date
    """

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame

def fetch_NASDAQ100_all_components():
    """Fetch NASDAQ100 all components data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.NASDAQ100.NASDAQ100_all_components` ORDER BY date
    """
    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame

def fetch_SP500_index():
    """Fetch SP500 index data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.SP500.SP500_index` ORDER BY date
    """

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame


def fetch_SP500_all_components():
    """Fetch SP500 all components data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.SP500.SP500_all_components` ORDER BY date
    """

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame

def fetch_ftse100_index():
    """Fetch FTSE 100 index data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.FTSE100.FTSE100_index` ORDER BY date
    """

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame

def fetch_ftse100_all_components():
    """Fetch FTSE 100 all components data from BigQuery and return as a Pandas DataFrame."""
    query = """
    SELECT * FROM `lewagon-statistical-arbitrage.FTSE100.FTSE100_all_components`ORDER BY date
    """

    # Initialize BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Convert results to Pandas DataFrame
    df = query_job.to_dataframe()

    return df  # Return the DataFrame
