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
fname = 'train.csv'
df = start.start('data/' + fname)
df = missing_values.fill_missing_values(df)
df = encoding.encode_nominal(df)
df = encoding.encode_ordinal(df)
df = feature_generation.generate(df)
df = feature_selection.select(df)
df = feature_transformations.transform(df)
df = anomaly_detection.remove(df)

#=========Summary======

print('=========Summary=========')
print("# of examples: {:} \n# of features: {:}".format(*df.shape))
print('Total # of missing values: ', df.isna().sum().sum())
print('Target: log_SalePrice')
print("Features: ", *df.columns.to_list(), "\n")
df.to_csv('data/preprocessed_' + fname)
print('Preprocessed data is saved to: data/preprocessed_' + fname)

