import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def svc_train(X_train, y_train, X_valid, y_valid):
    model = SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return model, valid_score


def svr_train(X_train, y_train, X_valid, y_valid):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return model, valid_score
