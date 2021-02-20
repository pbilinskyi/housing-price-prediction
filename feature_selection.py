import numpy as np
import pandas as pd


def select(df):
    df.drop(columns=['BsmtExposure', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 
                    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    '1stFlrSF', '2ndFlrSF', 
                    'LandContour', 'LandSlope', 'KitchenQual', 
                    'HeatingQC', 'ExterQual', 'ExterCond',
                    'BsmtQual', 'BsmtCond', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC', 
                    'WoodDeckSF', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'MSSubClass', 'DamageOfFunctionality',
                    'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 
                    'SaleCondition_AdjLand', 'SaleCondition_Family', 
                    'SaleCondition_Alloca', 'Foundation', 'LowQualFinSF',
                    'NearRailroad', 'MSZoning_FV', 'Fireplaces', 
                    'GarageCars', 'OverallCond', 'GarageFinish',
                    'SaleType'], inplace=True)
    # we will drop TotRmsAbvGrd, because 
    # it's redundant(highly correlated with GrLivArea) r = 0.82
    df.drop(columns=['TotRmsAbvGrd'], inplace=True)
    # if remodelling of house improves quality, OverallQual must
    # contain this information
    df.drop(columns=['YearRemodAdd'], inplace=True)
    df.drop(columns=['LotShape'], inplace=True)
    df.drop(columns=['BsmtFinType1', 'BsmtFinType2'], inplace=True)
    df.drop(columns=['Functional'], inplace=True)
    df.drop(columns=['Neighborhood'], inplace=True)
    print(f"# of selected features: {df.shape[1]}\n")
    return df
