import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def randomforest_train(X_train, y_train, X_valid, y_valid):
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return regr, valid_score
