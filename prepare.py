import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_polynomial(df):
    columns = ['OverallQual', 'log_GrLivArea',
               'GarageRating', 'YearBuilt',
               'FireplacesRating', 'KitchensRating', 
               'BsmtQuartersQ',
               'LotFrontage', 'LotArea']
    for feat in columns:
        for degree in [0.5, 2, 3]:
            poly_column_name = feat + ' ^ ' + str(degree)
            df.loc[:, poly_column_name] = df[feat]**degree
    print(f"Added polynomial terms of 0.5, 2, 3 degrees for {len(columns)} features\n")
    return df


def standard_scale(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print('Standard scaling: DONE\n')
    return X_train, X_test, scaler

