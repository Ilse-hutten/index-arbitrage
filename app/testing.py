import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from google.cloud import bigquery
import os


def z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, dynamic=False):

    stock_aligned = pca_weights_df[['date']].merge(underlying_df, on='date')
    weight_aligned = stock_aligned[["date"]].merge(pca_weights_df,on="date")
    weight_aligned.set_index("date",inplace=True)
    stock_aligned.set_index("date",inplace=True)

    for name in stock_aligned.columns:
        if name not in weight_aligned.columns:
            stock_aligned.drop(name,axis=1,inplace=True)

    target_df.index = pd.to_datetime(target_df.index)
    investment_aligned=weight_aligned.join(target_df)

    replication_aligned = weight_aligned.mul(investment_aligned['FTSE price'], axis=0)
    replication_aligned.sum(axis=1)

    weight_position = replication_aligned/stock_aligned

    #Calculating replication portfolio for cal_days+trade_days days based on Date PCA
    #
    replications_df=pd.DataFrame(columns=range(cal_days+trade_days), dtype=float)

    for i,r in weight_position.reset_index().iterrows():
        if i>cal_days:
            try_1 = r * stock_aligned[i-cal_days:min(i+trade_days, len(weight_position))]
            replication_index=pd.DataFrame(try_1.sum(axis=1).reset_index(drop=True))
            replications_df=pd.concat([replications_df,replication_index.T], axis=0)

    replications_df.index=weight_position.index[cal_days:-trading_length]
    replications_df.columns=[f'Day {i-cal_days+1}' for i in range(cal_days+trade_days)]
    replications_df=replications_df.astype(float)

    #Calculating target for cal_days+trade_days days from Date
    target_match_df=pd.DataFrame(columns=range(cal_days+trade_days))

    for i,r in weight_position.reset_index().iterrows():
        if i > cal_days:
            target_match=pd.DataFrame(target_df.iloc[i-cal_days:min(i+trade_days, len(weight_position))].reset_index(drop=True))
            target_match_df=pd.concat([target_match_df,target_match.T], axis=0)

    target_log_returns=np.log(target_match_df/target_match_df.shift(1, axis=1))
    replications_log_returns=np.log(replications_df/replications_df.shift(1, axis=1))

    spread_df=target_log_returns-replications_log_returns
    spread_mean=spread_df.iloc[:,:cal_days].mean(axis=1)
    spread_vol=spread_df.iloc[:,:cal_days].std(axis=1)

    z_scores_df=pd.DataFrame((spread_df.iloc[:,cal_days]-spread_mean)/spread_vol, columns=['z_score'])

    pos_low_threshold=0.5
    pos_high_threshold=2
    neg_low_threshold=-2
    neg_high_threshold=-0.5

    for i, r in z_scores_df.iterrows():
        if r['z_score'] > pos_low_threshold:
            if r['z_score'] <pos_high_threshold:
                z_scores_df.loc[i, 'direction']=-1
            else:
                z_scores_df.loc[i,'direction']=0
        elif r['z_score'] > neg_low_threshold:
            if r['z_score']<neg_high_threshold:
                z_scores_df.loc[i, 'direction']=1
            else: z_scores_df.loc[i, 'direction']=0

    return z_scores_df
