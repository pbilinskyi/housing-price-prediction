import numpy as np
import pandas as pd


def get_mask_isNA(df, column_name):
    return pd.isna(df[column_name])

def fill_missing_values(df):
    print('Total # of missing values: ', df.isna().sum().sum())
    lst = ['PoolQC', 'Fence', 'Alley', 'FireplaceQu', 'GarageFinish', 
           'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 
           'Electrical']
    df.loc[:, lst] = df[lst].fillna('No')

    #LotFrontage      259
    # determine missed frontage as mean among houses with the same number of cars
    mask_isNA = get_mask_isNA(df, 'LotFrontage')
    missed = df[mask_isNA]
    mean_frontages = (df[['LotFrontage', 'GarageCars']]
                      .groupby('GarageCars')
                      .mean()
                      .squeeze() 
                     )

    def fillna_frontage(r): 
        r.loc['LotFrontage'] = mean_frontages.loc[r['GarageCars']]
        return r
    
    df.loc[mask_isNA] = missed.apply(fillna_frontage, axis=1)
    print("Missing values inputation: DONE\n")
    return df
    