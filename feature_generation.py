import numpy as np
import pandas as pd


def generate(df):
    n_features_before = df.shape[1]
    means_neigh = pd.read_csv('data/mean_price_of_neighbours.csv',
                               index_col='Neighborhood')
    df = df.merge(means_neigh, 
                  how='left', 
                  left_on='Neighborhood', 
                  right_index=True)

    df['TotBath'] = df['BsmtFullBath'] + df['FullBath'] + 0.5*(df['BsmtHalfBath'] + df['HalfBath'])
    df['KitchensRating'] = df['KitchenAbvGr']*df['KitchenQual']
    df['FireplacesRating'] = df['Fireplaces']*df['FireplaceQu']
    df['GarageRating'] = df['GarageCars']*df['GarageQual']
    n_features_after = df.shape[1]
    print(f"# of generated features: {n_features_after - n_features_before}\n")
    return df

