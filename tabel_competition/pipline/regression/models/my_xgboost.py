import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10000
XGB_ROUND_NUM = 10000

objective = 'reg:squarederror'
metric = 'rmse'
params = {
    'objective': objective,
    'eval_metric': metric,
    'seed': 42,
    'tree_method': 'hist',
}
tuning_params = {
    "eta": 0.1,
    'lambda': 0,
    'max_depth': 3,
    "alpha": 0,
    "subsample": 1,
    "gamma": 0,
}


def xgboost_train(X_train, y_train, X_valid, y_valid):
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_valid_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(params, xgb_dataset,
                          XGB_ROUND_NUM,
                          evals=[(xgb_dataset, 'train'), (xgb_valid_dataset, 'eval')],
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose_eval=VERBOSE_EVAL)
    xgbm_va_pred = xgb_model.predict(xgb.DMatrix(X_valid))
    valid_score = np.sqrt(mean_squared_error(y_valid, xgbm_va_pred))
    return xgb_model, valid_score
