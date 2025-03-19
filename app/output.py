import seaborn as sns
import pandas as pd
import math

def output(bt_res):
    bt_results=bt_res.reset_index()
    bt_to_API=pd.DataFrame(bt_results['direction'] * (bt_results['target return']-bt_results['replication return']))
    bt_to_API.columns=['log return by trade']
    bt_to_API['direction']=bt_results['direction']

    bt_to_API['capital']=100
    bt_to_API['capital']=bt_to_API['capital'].astype(float)

    bt_to_API['log return by trade']=bt_to_API['log return by trade'].astype(float)
    bt_to_API['direction']=bt_to_API['direction'].astype(int)

    for i, row in bt_to_API.iterrows():
        bt_to_API.loc[i+1,'capital']=bt_to_API.loc[i,'capital']*math.exp(bt_to_API.loc[i,'log return by trade'])

    return bt_to_API

def saving_results():
    any_errors=False

    return any_errors
