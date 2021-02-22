import numpy as np
import pandas as pd


def transform(df):
    df.loc[:, 'log_GrLivArea'] = np.log(df['GrLivArea'])
    # TotalBsmtSD has 0 values, log1p() is used
    df.loc[:, 'log1_TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])

    df.drop(columns=['GrLivArea', 'TotalBsmtSF'])

    if 'SalePrice' in df.columns:
        df["log_SalePrice"] = np.log(df['SalePrice'])
        df.drop(columns=['SalePrice'], inplace=True)

    print("Log-transformations: DONE")
    return df
