import numpy as np
import pandas as pd

def remove(df):
    n_old = df.shape[0]
    if 'log_SalePrice' in df.columns.to_list():
        df = df.drop([1299, 31, 524, 534, 969, 1063, 496, 917, 1183, 692, 
                      873, 495, 741, 491, 411, 711, 637, 112, 1101])
    n_new = df.shape[0]
    print(f"Anomalies: removed {n_old - n_new} examples\n")
    return df
