import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from google.cloud import bigquery
import os


def z_score_trading(pca_weights_df, underlying_df, target_df, cal_days, trade_days, thresholds, exit_thresholds, dynamic=False):
    #
    #conventions: trade_days is the length of the trade with the initial day being counted as day 1
    #cal_days is the obeservation period of the spread with Day 0 being the Day 1 in the trade

    temp_df=pd.DataFrame(pca_weights_df.iloc[:,0])
    temp_df.columns=['temp']
    stock_aligned=pd.concat([temp_df,underlying_df], join='inner', axis=1)
    column_to_delete=stock_aligned.columns[0]
    stock_aligned.drop(column_to_delete, inplace=True, axis=1)

    temp_df=pd.DataFrame(stock_aligned.iloc[:,0])
    temp_df.columns=['temp']
    weight_aligned=pd.concat([temp_df,pca_weights_df], join='inner', axis=1)
    column_to_delete=weight_aligned.columns[0]
    weight_aligned.drop(column_to_delete, inplace=True, axis=1)

    temp_df=pd.DataFrame(stock_aligned.iloc[:,0])
    temp_df.columns=['temp']
    target_df.set_index("date",inplace=True)
    target_aligned=pd.concat([temp_df,target_df], join='inner', axis=1)
    column_to_delete=target_aligned.columns[0]
    target_aligned.drop(column_to_delete, inplace=True, axis=1)
    #

    replication_aligned = pd.DataFrame(weight_aligned.to_numpy() * target_aligned.to_numpy())
    weight_position = pd.DataFrame(replication_aligned.to_numpy()/stock_aligned.to_numpy())

    #reset of all indices
    weight_aligned=weight_aligned.reset_index()
    target_aligned=target_aligned.reset_index()
    stock_aligned=stock_aligned.reset_index()

    test_weight__aligned=weight_aligned.drop('date', axis=1)
    test_target__aligned=target_aligned.drop('date', axis=1)
    test_stock_aligned=stock_aligned.drop('date', axis=1)

    # Calculating replication portfolio for cal_days+trade_days days based on Date PCA

    replications_df=pd.DataFrame(columns=range(cal_days+trade_days), dtype=float)
    replications_df

    for i,r in weight_position.iterrows():
        if i>cal_days:
            combined = r.to_numpy() * test_stock_aligned[i-cal_days-1:min(i+trade_days-1, len(weight_position))].to_numpy()
            replication_index=pd.DataFrame(combined.sum(axis=1)).reset_index(drop=True)
            replications_df=pd.concat([replications_df,replication_index.T], axis=0)

    replications_df.index=weight_position.index[cal_days+1:]
    replications_df.columns=[f'Calibration Day {i-cal_days}' if i < (cal_days) else f'Trading Day {i-cal_days+1}' for i in range(0, cal_days+trade_days,1)]
    replications_df=replications_df.astype(float)

    # #Calculating target for cal_days+trade_days days from Date
    target_match_df=pd.DataFrame(columns=range(cal_days+trade_days,1))

    for i,r in weight_position.reset_index().iterrows():
        if i > cal_days:
            target_match=pd.DataFrame(target_aligned.iloc[i-cal_days-1:min(i+trade_days-1, len(weight_position)),1].reset_index(drop=True))
            target_match_df=pd.concat([target_match_df,target_match.T], axis=0)

    target_match_df.index=weight_position.index[cal_days+1:]
    target_match_df.columns=[f'Calibration Day {i-cal_days}' if i < (cal_days) else f'Trading Day {i-cal_days+1}' for i in range(0, cal_days+trade_days,1)]
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
    pos_low_threshold=thresholds[0]
    pos_high_threshold=thresholds[1]
    neg_low_threshold=thresholds[3]
    neg_high_threshold=thresholds[2]
    #
    for i, r in z_scores_df.iterrows():
        if r['z_score'] > pos_low_threshold:
            if r['z_score'] <pos_high_threshold:
                z_scores_df.loc[i, 'direction']=-1.0
            else:
                z_scores_df.loc[i,'direction']=0.0
        elif r['z_score'] > neg_low_threshold:
            if r['z_score']<neg_high_threshold:
                z_scores_df.loc[i, 'direction']=1.0
            else: z_scores_df.loc[i, 'direction']=0.0
        else: z_scores_df.loc[i, 'direction']=0.0
    #
    #spread from today to the end
    #
    spread_df=pd.DataFrame(target_log_returns.iloc[:,cal_days+1:].values-replications_log_returns.iloc[:,cal_days+1:].values, index=target_log_returns.index)
    #
    #
    z_scores_df['target entry']=target_match_df['Trading Day 1']
    z_scores_df['replication entry']=replications_df['Trading Day 1']
    #
    for day in np.arange(0,trade_days-1, 1):
        single_score_df=pd.DataFrame(((spread_df.iloc[:,day]-spread_mean)/spread_vol), columns=[f'Day {day+1}'])
        z_scores_df=pd.concat((z_scores_df,single_score_df), axis=1)
    #
    #
    #
    for i, row in z_scores_df.iterrows():
        #test_date=pd.to_datetime(start_date).strftime('%Y-%m-%d')
        #
        #default setting to maximum trade days
        exit_day=trade_days-1

        #if the setting is dynamic looping through the z scores to identify when the position would have been closed
        if dynamic:
            if row['direction']==1:
                for day in range(2, trade_days-1):
                    if row[f'Day {day}'] > exit_thresholds[0]:
                        exit_day=day
                        break
            if row['direction']==-1:
                for day in range(2, trade_days-1):
                    if row[f'Day {day}'] < exit_thresholds[1]:
                        exit_day=day
                        break

        z_scores_df.loc[i, 'replication exit']=replications_df.loc[i, f'Trading Day {exit_day}']
        z_scores_df.loc[i, 'target exit']=target_match_df.loc[i, f'Trading Day {exit_day}']

    z_scores_df['target return']=np.log(z_scores_df['target exit']/z_scores_df['target entry'])
    z_scores_df['replication return']=np.log(z_scores_df['replication exit']/z_scores_df['replication entry'])

    print(z_scores_df)

    return z_scores_df
