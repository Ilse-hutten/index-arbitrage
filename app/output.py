import seaborn as sns
import pandas as pd
import math

def output(bt_results):
    bt_to_API=pd.DataFrame(bt_result['direction'] * (bt_result['target return']-bt_result['replication return']))
    bt_to_API.columns=['log return by trade']
    pd.concat(bt_to_API, bt_result['direction'])


    bt_to_API['capital']=100
    for i, row in bt_to_API.iterrows():
        bt_to_API.loc[i+1,'capital']=bt_to_API.loc[i,'capital']*math.exp(bt_to_API.loc[i,'log return by trade'])

    return bt_to_API

def saving_results():
    any_errors=False

    return any_errors
