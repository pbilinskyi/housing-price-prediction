#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import start
import missing_values
import encoding
import feature_generation
import feature_selection
import feature_transformations
import anomaly_detection


pd.set_option('display.max_rows', 10)
pd.set_option("display.precision", 2)

#=========MAIN=========

def preprocess(fname):
    print('=================================')
    is_training_set = 'train' in fname

    df = start.start('data/' + fname)
    df = missing_values.fill_missing_values(df)
    df = encoding.encode_nominal(df)
    df = encoding.encode_ordinal(df)
    df = feature_generation.generate(df)
    df = feature_selection.select(df)
    df = feature_transformations.transform(df)
    if is_training_set: 
        df = anomaly_detection.remove(df)
    df.to_csv('data/preprocessed_' + fname)

    if not is_training_set:
        df = df.fillna(0)

    print('==Summary==')
    print("# of examples: {:} \n# of features: {:}".format(*df.shape))
    print('Total # of missing values: ', df.isna().sum().sum())
    if 'train' in fname:
        print('Target: log_SalePrice')
    print("Features: ", *df.columns.to_list(), "\n")
    print('Preprocessed data is saved to: data/preprocessed_' + fname)


preprocess('train.csv')
preprocess('test.csv')