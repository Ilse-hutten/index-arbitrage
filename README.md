# stock-stat-replica
General Structure of the project


Data
      Project folder needed, which contains the data to be used in the analysis
      in csvs.

      data to be stored in the subfolder "/data/" in the general project folder

      csvs need to have the format of
      date ('yyy-mm-dd')
      open
      high
      low
      close
      volume

      Provides dataframe to model section
      Initial starting from Feb 22 to today()

Model

  returns= daily log returns

  PCA selects the eigenvectors which explain at least 90% of the variance
  top X stocks are selected by weight in the sum of the eigenvectors

Test


Backtest
