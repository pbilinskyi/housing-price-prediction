# Changelog

## [1.1.2] - 2021-02-22

### Added

- Project gets even more structured. I splitted training and predicting process into the following modules:
    1. [Main module](train_main.py)
    2. [Preparation: scaling and adding polynomial terms](prepare.py)
    3. [Returning fitted model](fit_model.py)
    4. [Measuring performance](measure.py)

   [Main module](train_main.py) launches the whole process.

## [1.1.1] - 2021-02-20

### Added

- Project gets more structured. I splitted data preprocessing into the following modules:
    1. [Starting](start.py)
    2. [Missing values handling](missing_values.py)
    3. [Feature encoding](encoding.py)
    4. [Feature generation](feature_generation.py)
    5. [Feature selection](feature_selection.py)
    6. [Feature transformations](feature_transformations.py)
    7. [Anomaly detection](anomaly_detection.py)

   [main module](main.py) gathers them together.


## [1.1.0] - 2021-02-19

### TODO

- Perform feature selection with LASSO or XGBoost
- Plot *learning curve* to detect underfitting or overfitting
