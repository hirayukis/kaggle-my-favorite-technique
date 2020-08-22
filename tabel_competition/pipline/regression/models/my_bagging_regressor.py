import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error


def bagging_regressor_train(X_train, y_train, X_valid, y_valid):
    model = BaggingRegressor(base_estimator=SVR(),
                             n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return model, valid_score
