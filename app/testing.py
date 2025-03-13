import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from google.cloud import bigquery
from google.colab import files
import os

uploaded = files.upload()  # Manually upload the JSON key file

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCPCredentials

client = bigquery.Client()

# Set your GCP project details
project_id = PROJECTID

#SUGGESTION need to make this flexible
dataset_id = "FTSE_100_main"  # Replace with your dataset name
table_name = "FTSE100_csv"

# Construct full table path
table_id = f"{project_id}.{dataset_id}.{table_name}"

#query BigQuery
query = f"SELECT * FROM `{table_id}` ORDER BY Unnamed_0 ASC"

target_df = client.query(query).to_dataframe()

target_df.rename(columns={'Unnamed_0': 'Date',"close": "FTSE price"}, inplace=True)
target_df.set_index('Date', inplace=True)
target_close_price_df = pd.DataFrame(FTSE100["FTSE price"])

#reading statistical analysis output
daily_weight = pd.read_csv("daily_weights.csv")

#removing .L ending for London denoted stocks
daily_weight = daily_weight.rename(columns = lambda x : str(x)[:-2])
daily_weight = daily_weight.rename(columns={'Da': 'date'})
daily_weight["date"] = pd.to_datetime(daily_weight["date"])

#setting the data onto a single dataframe for frequency and date and stock match
stock_aligned = daily_weight[["date"]].merge(stock_price,on="date")
weight_aligned = stock_aligned[["date"]].merge(daily_weight,on="date")
weight_aligned.set_index("date",inplace=True)
stock_aligned.set_index("date",inplace=True)

for name in stock_aligned.columns:
  if name not in weight_aligned.columns:
    stock_aligned.drop(name,axis=1,inplace=True)

    #dropping BTA from the stock index dataframe
    weight_aligned = weight_aligned.drop("BT-A", axis=1)

    #test that the shapes are the same between the weight and the stock dataframe

#converting
target_close_price_df.index = pd.to_datetime(target_close_price_df.index)

