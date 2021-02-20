import numpy as np
import pandas as pd


def transform(df):
    if 'SalePrice' in df.columns:
        df["log_SalePrice"] = np.log(df['SalePrice'])

    df.loc[:, 'log_GrLivArea'] = np.log(df['GrLivArea'])
    # TotalBsmtSD has 0 values, log1p() is used
    df.loc[:, 'log1_TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])
    df = df.fillna(0)
    print("Log-transformations: DONE")
    return df
