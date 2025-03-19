import datetime
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from data_querry  import  *
import pandas as pd
app = FastAPI()



@app.get('/')
def index():
    return {'ok': True}

def clean_dataframe(df):
    for col in df.select_dtypes(include=['datetime64', 'object']):
        df[col] = df[col].apply(lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime)) else x)

    df.fillna(method='ffill', inplace=True)
    return df

@app.get('/dataset_name')
def get_dataset_name(name: str):
    if name == 'NASDAQ100':
        index_df = fetch_NASDAQ100_index()
        comp_df = fetch_NASDAQ100_all_components()

        index_json = json.loads(index_df.to_json(orient="records", date_format="iso"))
        comp_json = json.loads(comp_df.to_json(orient="records", date_format="iso"))

        data = {
            "index": index_json,
            "comp": comp_json
        }
        return JSONResponse(content=data)

    elif name == 'SP500':
        index_df = fetch_SP500_index()
        comp_df = fetch_SP500_all_components()

        index_json = json.loads(index_df.to_json(orient="records", date_format="iso"))
        comp_json = json.loads(comp_df.to_json(orient="records", date_format="iso"))

        data = {
            "index": index_json,
            "comp": comp_json
        }
        return JSONResponse(content=data)

    elif name == 'FTSE100':
        index_df = fetch_ftse100_index()
        comp_df = fetch_ftse100_all_components()

        index_json = json.loads(index_df.to_json(orient="records", date_format="iso"))
        comp_json = json.loads(comp_df.to_json(orient="records", date_format="iso"))

        data = {
            "index": index_json,
            "comp": comp_json
        }
        return JSONResponse(content=data)

    else:
        return JSONResponse(content={"error": "Invalid dataset name."}, status_code=400)


# @app.post('/get_pca')
# def get_pca(days,time,n_stocks):
#     return {}

# @app.get('/test_bq')
# def get_pca():
#     index_df = fetch_ftse100_index()
#     index_json = json.loads(index_df.to_json(orient="records", date_format="iso"))

#     return {"index": index_json}
