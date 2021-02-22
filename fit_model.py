import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from xgboost import XGBRegressor


optimal_params = {'colsample_bytree': 0.5,
                'learning_rate': 0.1,
                'max_depth': 10,
                'min_child_weight': 1,
                'n_estimators': 200,
                'objective': 'reg:squarederror',
                'subsample': 1}

def fit(X, y):
    xgbr = XGBRegressor(**optimal_params)
    xgbr.fit(X, y, verbose=0)

    # importances = pd.Series(xgbr.feature_importances_, index=features)
    # importances = importances.sort_values(ascending=False)
    return xgbr