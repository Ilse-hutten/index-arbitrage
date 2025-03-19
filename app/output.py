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

def alternative_asset_return(bt_res, alt_dev):
    alt_return_df=pd.DataFrame(bt_res['direction'])
    alt_return_df=alt_return_df.join(bt_res[['target entry']])
    alt_return_df['daily target return']=np.log(alt_return_df['target entry']/alt_return_df['target entry'].shift(1))
    alt_return_df['excess return']=alt_return_df['direction']*(bt_res['target return']-bt_res['replication return'])

    alt_return_df=alt_return_df.reset_index()

    alt_return_df['strategy']=alt_return_df['target entry']

    for i in range(1, len(alt_return_df)):
        if alt_return_df.iloc[i]['direction'] == 0:
            alt_return_df.loc[i, 'strategy'] = alt_return_df.loc[i-1, 'strategy'] * math.exp(alt_return_df.loc[i, 'daily target return'])
        else:
            alt_return_df.loc[i, 'strategy'] = alt_return_df.loc[i-1, 'strategy'] * math.exp(alt_return_df.loc[i, 'excess return'])

    return alt_return_df
