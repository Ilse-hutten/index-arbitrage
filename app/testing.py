import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from google.cloud import bigquery
import os


def z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, thresholds, dynamic=False):

    #conventions: trade_days is the length of the trade with the initial day being counted as day 1
    #cal_days is the obeservation period of the spread with Day 0 being the Day 1 in the trade

    stock_aligned = pca_weights_df[['date']].merge(underlying_df, on='date')
    weight_aligned = stock_aligned[["date"]].merge(pca_weights_df,on="date")
    weight_aligned.set_index("date",inplace=True)
    stock_aligned.set_index("date",inplace=True)

    for name in stock_aligned.columns:
        if name not in weight_aligned.columns:
            stock_aligned.drop(name,axis=1,inplace=True)

    #target_df.index = pd.to_datetime(target_df.index)

    investment_aligned=weight_aligned.join(target_df)
    #
    #modify for target name!!!
    target_df = investment_aligned['FTSE price']

    #need to make this flexible here!! FTSE price
    #
    replication_aligned = weight_aligned.mul(investment_aligned['FTSE price'], axis=0)
    weight_position = replication_aligned/stock_aligned

    # #Calculating replication portfolio for cal_days+trade_days days based on Date PCA
    # #
    replications_df=pd.DataFrame(columns=range(cal_days+trade_days), dtype=float)

    for i,r in weight_position.reset_index().iterrows():
        if i>cal_days:
            try_1 = r * stock_aligned[i-cal_days-1:min(i+trade_days-1, len(weight_position))]
            replication_index=pd.DataFrame(try_1.sum(axis=1).reset_index(drop=True))
            replications_df=pd.concat([replications_df,replication_index.T], axis=0)

    replications_df.index=weight_position.index[cal_days+1:]
    replications_df.columns=[f'Calibration Day {i-cal_days}' if i < (cal_days-1) else f'Trading Day {i-cal_days+1}' for i in range(0, cal_days+trade_days,1)]
    replications_df=replications_df.astype(float)

    # #Calculating target for cal_days+trade_days days from Date
    target_match_df=pd.DataFrame(columns=range(cal_days+trade_days,1))

    for i,r in weight_position.reset_index().iterrows():
        if i > cal_days:
            target_match=pd.DataFrame(target_df.iloc[i-cal_days-1:min(i+trade_days-1, len(weight_position))].reset_index(drop=True))
            target_match_df=pd.concat([target_match_df,target_match.T], axis=0)

    target_match_df.index=weight_position.index[cal_days+1:]
    target_match_df.columns=[f'Calibration Day {i-cal_days}' if i < (cal_days-1) else f'Trading Day {i-cal_days+1}' for i in range(0, cal_days+trade_days,1)]
    target_match_df=target_match_df.astype(float)

    #log_return calculation of the target and replication
    #
    target_log_returns=np.log(target_match_df/target_match_df.shift(1, axis=1))
    replications_log_returns=np.log(replications_df/replications_df.shift(1, axis=1))

    #calculating difference in log returns over the last 60 days to today
    #
    spread_past_df=pd.DataFrame(target_log_returns.iloc[:,1:cal_days+1].values-replications_log_returns.iloc[:,1:cal_days+1].values, index=target_log_returns.index)
    spread_mean=spread_past_df.mean(axis=1)
    spread_vol=spread_past_df.std(axis=1)

    #todays spread's z-score
    #
    z_scores_df=pd.DataFrame((spread_past_df.iloc[:,-1]-spread_mean)/spread_vol, columns=['z_score'])

    #checking the trading signal against thresholds
    #
    pos_low_threshold=0.5
    pos_high_threshold=2
    neg_low_threshold=-2
    neg_high_threshold=-0.5
    #
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

    #spread from today to the end
    #
    spread_df=pd.DataFrame(target_log_returns.iloc[:,cal_days+1:].values-replications_log_returns.iloc[:,cal_days+1:].values, index=target_log_returns.index)

    z_scores_df['target entry']=target_df
    z_scores_df['replication entry']=replications_df['Trading Day 1']

    for day in np.arange(0,trade_days-1, 1):
        single_score_df=pd.DataFrame(((spread_df.iloc[:,day]-spread_mean)/spread_vol), columns=[f'Day {day+1}'])
        z_scores_df=pd.concat((z_scores_df,single_score_df), axis=1)

    print(z_scores_df)


    for start_date, row in z_scores_df.iterrows():
        test_date=pd.to_datetime(start_date).strftime('%Y-%m-%d')
        #
        #default setting to maximum trade days
        exit_day=trade_days-1

        #if the setting is dynamic looping through the z scores to identify when the position would have been closed
        if dynamic:
            if row['direction']==1:
                for day in range(1, trade_days-1):
                    if row[f'Trading Day {day}'] > exit_threshold:
                        exit_day=day
                        break
            if row['direction']==-1:
                for day in range(1, trade_days-1):
                    if row[f'Trading Day {day}'] < exit_threshold:
                        exit_day=day
                        break
        z_scores_df.loc[test_date, 'replication exit']=replications_df.loc[test_date, f'Trading Day {exit_day}']
        z_scores_df.loc[test_date, 'target exit']=target_match_df.loc[test_date, f'Trading Day {exit_day}']

    z_scores_df['target return']=np.log(z_scores_df['target exit']/z_scores_df['target entry'])
    z_scores_df['replication return']=np.log(z_scores_df['replication exit']/z_scores_df['replication entry'])

    return z_scores_df
