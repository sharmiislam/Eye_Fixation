
# models/regression.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


class RegressionEstimators:
    def __init__(self, estimator, params):
        self.estimator = estimator
        self.params = params

    def tune_hyperparameters(self, X_train, y_train):
        # Perform hyperparameter tuning
        pass

    def train_test_estimator(self, X_train, X_test, y_train, y_test):
        # Train and test the estimator
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse


# Additional methods can be added for saving/loading models, printing results, etc.

