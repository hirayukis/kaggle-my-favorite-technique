import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression_train(X_train, y_train, X_valid, y_valid):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return model, valid_score
