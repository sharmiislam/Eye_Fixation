# models/model_evaluation.py
from sklearn.metrics import mean_squared_error

class ModelEvaluation:
    def __init__(self, models):
        self.models = models
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{model_name} MSE: {mse}")
