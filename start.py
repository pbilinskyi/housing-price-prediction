#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def start(fname):
    is_training = 'train' in fname
    print("\n We are processing ", end='')
    if is_training:
        print("TRAIN set\n")
    else:
        print("TEST set\n")

    df = pd.read_csv(fname, index_col='Id')
    # print("Target: SalePrice\n")
    print('Features: \n', *df.columns.to_list(), '\n')
    print("# of examples: {:} \n# of features: {:}\n".format(*df.shape))
    
    # Removing features we're not going to use
    lst = ['MoSold', 'YrSold', 'BldgType',     # not a useful information
           'MasVnrType', 'MasVnrArea',         # not a useful information: nobody cares about masonry
           'Heating', 'RoofStyle', 'RoofMatl',
           'Exterior1st', 'Exterior2nd', 'LotConfig',
           'GarageArea',                       # redundant, highly correlated with GarageCars (r ~ 0.88)
           'GarageYrBlt',                      # redundant (almost always == YearBuilt)
           'GarageType', 'HouseStyle', 
           'MiscFeature', 'MiscVal']
    df.drop(columns=lst, inplace=True)
    return df
