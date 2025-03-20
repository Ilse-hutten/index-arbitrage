import datetime
import json
from typing import List
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from data_query import  *
import pandas as pd
from main import compute_bt_result
app = FastAPI()




@app.get('/')
def index():
    return {'ok': True}

def clean_dataframe(df):
    for col in df.select_dtypes(include=['datetime64', 'object']):
        df[col] = df[col].apply(lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime)) else x)

    #df.fillna(method='ffill', inplace=True)
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


@app.get('/fetch_btresult_rolling_pca')
def fetch_btresult_rolling_pca(
    cal_days:int,
    trade_days: int,
    n_stocks:int,
    window:int,
    n_pcs:int,
    thresholds:List[float] = Query([2, 200, -2, -200],description="Send in the order."),
    exit_levels: List[float] =[-0.5,0.5],
    index_selected:str='SP500',

    ):

    bt_result,rep_pf=compute_bt_result(cal_days,trade_days,n_stocks,window,
    n_pcs,thresholds,index_selected,exit_levels,True)
    rep_pf.reset_index(inplace=True)
    data={
            "bt_result": json.loads(bt_result.to_json(orient="records", date_format="iso")),
            "rep_pf":  json.loads(rep_pf.to_json(orient="records", date_format="iso"))

        }
    print(data)
    return JSONResponse(content=data)


# print(compute_bt_result(30,30,30,30,4,[0.5, 2, -0.5, -2],'SP500'))
