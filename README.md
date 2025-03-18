# stock-stat-replica
General Structure of the project


Data
      Project folder needed, which contains the data to be used in the analysis
      in csvs.

      data to be stored in the subfolder "/data/" in the general project folder

      csvs need to have the format of
      * data format may need to change in further testing *


      date ('yyyy-mm-dd')

      rename_dict = {
                        'Unnamed: 0': 'date',
                        '1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low',
                        '4. close': 'close',
                        '5. volume': 'volume'
                    }

      Provides dataframe to model section
      Initial starting from Feb 22 to today()

      Output:
        dataframe:
          index: date
          column: stocks used to perform PCA

Model

  Expects:
    dataframe:
            index: date
            column: stocks used to perform PCA

  Calculates:
            returns= log returns based on the frequency of the fed dataframe

  PCA selects the eigenvectors which explain at least 90% of the variance
  top X stocks are selected by weight in the sum of the eigenvectors

Test
  Expects: Dataframe with
            rows:     Date
            columns:  All stocks used in the replication portfolio
            values:   weight of the respective stock on the Date

  Creates:
            Replication portfolio by using Weight Dataframe, stock prices and target asset (e.g. FTSE 100) loaded from the data folder.

            Drops all stock prices and dates not present in the weight dataframe

            Establishes "positions" depending on a minimum z score and maximum z score

            *fine tune trading strategy*

Backtest
