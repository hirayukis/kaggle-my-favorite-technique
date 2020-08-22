import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def ridge_train(X_train, y_train, X_valid, y_valid):
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return model, valid_score
