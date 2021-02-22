import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def rmse_cv(model, X, y, agg='mean'):
    RMSEs = np.sqrt(-cross_val_score(model, X, y, 
                     scoring="neg_mean_squared_error"))
    if agg=='mean':
        return RMSEs.mean()
    else:
        return RMSEs
