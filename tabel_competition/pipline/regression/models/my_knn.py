import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def knnr_train(X_train, y_train, X_valid, y_valid):
    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return neigh, valid_score
