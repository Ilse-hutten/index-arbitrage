from google.cloud import storage
import pandas as pd
import io
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Frame:
    def dataset(self,name='FTSE_100'):
        print('Fetching and Merging...')
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../data/" + 'lewagon-statistical-arbitrage-ae470f7dcd48.json'
        client = storage.Client()
        bucket = client.get_bucket('stat_arb')
        rename_dict = {
                        'Unnamed: 0': 'date',
                        '1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low',
                        '4. close': 'close',
                        '5. volume': 'volume'
                    }
        expected_columns = list(rename_dict.values())

        #SUGGESTION Folder prefix should be made flexible eventually
        folder_prefix = name+"/"


        blobs = bucket.list_blobs(prefix=folder_prefix)
        dataframes = []

        for blob in blobs:
            if blob.name.endswith('.csv'):
                content = blob.download_as_string()
                if content.strip():
                    try:
                        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=None, engine='python')
                        df = df.rename(columns=rename_dict)
                        df = df.loc[:, df.columns.intersection(expected_columns)]
                        stk=blob.name.split('/')[1]
                        df['source_file']=stk.split('.')[0]
                        dataframes.append(df)

                    except pd.errors.EmptyDataError:
                        print(f"Skipping empty file: {blob.name}")
                else:
                    print(f"Skipping empty file: {blob.name}")

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            print("Merging completed successfully!")
        else:
            print("No valid CSV files found.")
        print('Started prepocessing....')
        merged_df['date'] = pd.to_datetime(merged_df['date'],format='%Y-%m-%d')

        df_modified = merged_df[['date', 'source_file', 'close']].copy()
        df_modified['source_file'] = df_modified['source_file'].str.split('/').str[-1].str.replace('.csv', '')
        df_pivoted = df_modified.pivot_table(index='date', columns='source_file', values='close')
        df_pivoted.reset_index(inplace=True)

        # df_latest=df_pivoted[df_pivoted['date']>'2022-01-31']
        df_pivoted.fillna(method='ffill', inplace=True)
        print('Completed prepocessing.')
<<<<<<< HEAD
        return df_latest
=======
        return df_pivoted
>>>>>>> 8cc847859cb5869f40e47d54b507f70e7f9aa3a1
