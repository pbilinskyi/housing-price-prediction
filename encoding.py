import numpy as np
import pandas as pd


def get_dummies_if_any(df, lst):
    return  ((np.isin(df['Condition1'], lst) |
              np.isin(df['Condition2'], lst))
                     .astype(int)
            )


def encode_nominal(df):
    # # OneHot encoding
    df = df.merge(pd.get_dummies(df[['SaleCondition', 'MSZoning']]),
                left_index=True, 
                right_index=True, 
                how='left')
    df.drop(columns=['SaleCondition', 'MSZoning'], inplace=True)

    # Making binary:
    df['NearArtery_or_Feedr'] = get_dummies_if_any(df, ['Artery', 'Feedr'])
    df['PositiveFeat'] = get_dummies_if_any(df, ['PosA', 'PosN'])
    df['NearRailroad'] = get_dummies_if_any(df, ['RRAe', 'RRAn', 'RRNe', 'RRNn'])
    df.drop(columns=['Condition1','Condition2'], inplace=True)

    df['Foundation'] = (df['Foundation'] != 'Wood').astype(int)

    df['RegularityOfShape'] = ((df['LotShape'] == 'Reg') | (df['LotShape'] == 'IR1')).astype(int)
    
    print("Encoding nominal features: DONE")
    return df

def encode_ordinal(df):
    # # Ordinal encoding

    lst = ['PoolQC', 'KitchenQual', 
        'HeatingQC', 'ExterQual', 'ExterCond',
        'BsmtQual', 'BsmtCond', 'FireplaceQu',
        'GarageQual', 'GarageCond']
    quality_map={'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df.loc[:, lst] = df[lst].apply(lambda c: c.map(quality_map), axis=0).fillna(0)


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
    print("Encoding ordinal features: DONE\n")
    return df



