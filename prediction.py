import numpy as np
import pandas as pd

def predict_and_save(model, df):
    X = df.to_numpy()
    y_pred = model.predict(X)
    y_pred = pd.Series(np.exp(y_pred), index=df.index)
    y_pred = y_pred.rename("SalePrice")
    y_pred.to_csv('submission.csv')
    return