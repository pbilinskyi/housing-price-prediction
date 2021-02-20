# Housing Prices Prediction

Predictive model for competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) on Kaggle.

## Pipeline of the model

Consider the following scheme:
<p align="center">
   <img src="images/pipeline.svg" />
</p>

1. Preprocessing.
   - cleaning
   - missing values handling
   - feature encoding
   - feature generation
   - feature selection

2. Skewness; outliers detection.
   - basic Exploratory Data Analysis
   - transformations of features
   - outliers detection and removing

3. Model selection and validation.
   - cross-validation
   - hyperparameters tuning
   - training ML models: Ridge, LASSO, ElasticNet, Support Vector Regressor, XGBoost Regressor
   - computing feature's importances
   - prediction

## Achievments
Position on the leaderboard: top 47%.

![Leaderboard](images/leaderboard.jpeg)
