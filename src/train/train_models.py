from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    # Instantiate the Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model