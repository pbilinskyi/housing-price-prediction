#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 90)
pd.set_option("display.precision", 2)

fname = input('filename: ')
is_training = 'train' in fname

print("We are processing ", end='')
if is_training:
    print("TRAIN set")
else:
    print("TEST set")

df = pd.read_csv(fname, index_col='Id')
print('Features: \n', df.columns)

if is_training:
    y = df['SalePrice']
    df = df.drop(columns='SalePrice')
n_examples, n_features = df.shape



# # Getting rid of not interesting features
lst = ['MoSold', 'YrSold', 'BldgType',      # not so useful information
       'MasVnrType', 'MasVnrArea',         # nobody cares about masonry
       'Heating', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'LotConfig',
       'GarageArea',                       # redundant, highly correlated with GarageCars (r ~ 0.88)
       'GarageYrBlt',                      # redundant (almost always == YearBuilt)
       'GarageType', 'HouseStyle', 
       'MiscFeature', 'MiscVal']
df.drop(columns=lst, inplace=True)


# # Missing values inputation 
def get_mask_isNA(column_name):
    return pd.isna(df[column_name])


lst = ['PoolQC', 'Fence', 'Alley', 'FireplaceQu', 'GarageFinish', 
       'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 
       'Electrical']
df.loc[:, lst] = df[lst].fillna('No')


#LotFrontage      259
# determine missed frontage as mean among houses with the same number of cars
mask_isNA = get_mask_isNA('LotFrontage')
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


# # OneHot encoding
df = df.merge(pd.get_dummies(df[['SaleCondition', 'MSZoning']]),
              left_index=True, 
              right_index=True, 
              how='left')
df.drop(columns=['SaleCondition', 'MSZoning'], inplace=True)

# Special case:
# 3 one-hot vectors from features Condition1 and Condition2:
# 
# first: Artery  or Feedr is present
# 
# second: PosA or PosN
# 
# third: RRAe or RRAn or RRNe or RRNn

def get_dummies_if_any(df, lst):
    return  ((np.isin(df['Condition1'], lst) |
              np.isin(df['Condition2'], lst))
                     .astype(int)
            )

df['NearArtery_or_Feedr'] = get_dummies_if_any(df, ['Artery', 'Feedr'])
df['PositiveFeat'] = get_dummies_if_any(df, ['PosA', 'PosN'])
df['NearRailroad'] = get_dummies_if_any(df, ['RRAe', 'RRAn', 'RRNe', 'RRNn'])
df.drop(columns=['Condition1','Condition2'], inplace=True)


# # Ordinal encoding

lst = ['PoolQC', 'KitchenQual', 
       'HeatingQC', 'ExterQual', 'ExterCond',
       'BsmtQual', 'BsmtCond', 'FireplaceQu',
       'GarageQual', 'GarageCond']
quality_map={'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df.loc[:, lst] = df[lst].apply(lambda c: c.map(quality_map), axis=0).fillna(0)

#================================================
def get_quality_coef(column_name, grades_to_numbers):
    return df[column_name].map(grades_to_numbers)

# mask_isNA = df['Fence'] == 'No'
# df.loc[~mask_isNA, 'Fence'] = 1
# df.loc[mask_isNA, 'Fence'] = 0
df['Fence'] = get_quality_coef('Fence', 
                               {'No': 0}).fillna(1)
# df['BsmtExposure'] = get_quality_coef('BsmtExposure', 
#                                       {'No': 1, 'Mn': 2, 'Av': 3, 'Ex' : 4}).fillna(0)
df['BsmtQuartersQ'] = get_quality_coef('BsmtFinType1', 
                                       {'BLQ':1, 'ALQ':2, 'GLQ':3}).fillna(0) +\
                      get_quality_coef('BsmtFinType2', 
                                       {'BLQ':1, 'ALQ':2, 'GLQ':3}).fillna(0)
df['CentralAir'] = df['CentralAir'].map({'N':0, 'Y':1})
df['Electrical'] = get_quality_coef('Electrical', {'FuseP':1}).fillna(0)
# df['GarageFinish'] = get_quality_coef('GarageFinish', 
#                                       {'Fin':3, 'RFn':2, 'Unf':1}).fillna(0)
df['PavedDrive'] = get_quality_coef('PavedDrive', 
                                    {'N':1, 'P':2, 'Y':3})
df['Utilities'] = get_quality_coef('Utilities', 
                                   {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}).fillna(0)
# df['LandContour'] = get_quality_coef('LandContour', 
#                                      {'Low':3, 'Bnk':2, 'Hls':1}).fillna(0)
# df['LandSlope'] =  get_quality_coef('LandSlope', 
#                                     {'Gtl':1, 'Mod':2, 'Sev':3})
df['DeductionsFunctional'] = get_quality_coef('Functional', 
                                              {'Typ':0, 'Min1':1, 'Min2':2,
                                               'Mod':3, 'Maj1':4, 'Maj2':5}).fillna(0)
df['DamageOfFunctionality'] = get_quality_coef('Functional', 
                                               {'Sev':1, 'Sal': 2}).fillna(0)
df['Street'] = get_quality_coef('Street', {'Grvl':1, 'Pave':2})
df['Alley'] = get_quality_coef('Alley', {'Grvl':1, 'Pave':2}).fillna(0)
df['Foundation'] = (df['Foundation'] != 'Wood').astype(int)

# LotShape regularity - binary feature
df['RegularityOfShape'] = ((df['LotShape'] == 'Reg') | (df['LotShape'] == 'IR1')).astype(int)



# # Generate new features

means_neigh = pd.read_csv('mean_price_of_neighbours.csv',
                           index_col='Neighborhood')
df = df.merge(means_neigh, how='left', left_on='Neighborhood', right_index=True)

df['TotBath'] = df['BsmtFullBath'] + df['FullBath'] + 0.5*(df['BsmtHalfBath'] + df['HalfBath'])
df['KitchensRating'] = df['KitchenAbvGr']*df['KitchenQual']
df['FireplacesRating'] = df['Fireplaces']*df['FireplaceQu']
df['GarageRating'] = df['GarageCars']*df['GarageQual']


# # Dropping of non-useful features
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


# # Log-transformation and anomalies removing
if is_training:
    y = np.log(y)
    df["log_SalePrice"] = y

df.loc[:, 'log_GrLivArea'] = np.log(df['GrLivArea'])
df.loc[:, 'log1_TotalBsmtSF'] = df['TotalBsmtSF'].transform(lambda x: np.log(1 + x))
print(f"missing values: {df.isna().sum().sum()}")
df = df.fillna(0)

# # Anomalies removing
if is_training:
    df = df.drop([1299, 31, 524, 534, 969, 1063, 496, 917, 1183, 692, 
                  873, 495, 741, 491, 411, 711, 637, 112, 1101])


# # Summary

print('======Summary======')
print('examples = ', df.shape[0])
print('features = ', df.shape[1])
print('Total # of missing values: ', df.isna().sum().sum())

print(df.columns)
df.to_csv('preprocessed_' + fname)
print('preprocessed data is saved to: ', 'preprocessed_' + fname)
