import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import prepare
import fit_model
import measure
import prediction


fname = 'train.csv'
df = pd.read_csv('data/preprocessed_' + fname, index_col='Id')
X, y = df.drop(columns=['log_SalePrice']), df['log_SalePrice']

X = prepare.generate_polynomial(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
X_train, X_valid, scaler = prepare.standard_scale(X_train, X_valid)

model = fit_model.fit(X_train, y_train)
err_train = measure.rmse_cv(model, X_train, y_train, agg='mean')
err_valid = measure.rmse_cv(model, X_valid, y_valid, agg='mean')
print(f'err. on training set: {err_train}')
print(f'err. on validation set: {err_valid}\n')

#=======Submission generation==================
print('======Predictions generation for test set======')
fname = 'test.csv'
df_test = pd.read_csv('data/preprocessed_' + fname, index_col='Id')
columns, index = df_test.columns, df_test.index
df_test = prepare.generate_polynomial(df_test)
X_test = scaler.transform(df_test)
print('Standard scalind: DONE')
y_pred = model.predict(X_test)

y_pred = pd.Series(np.exp(y_pred), index=index)
y_pred = y_pred.rename("SalePrice")
y_pred.to_csv('data/submission.csv')
print('Predictions were saved to data/submission.csv')
