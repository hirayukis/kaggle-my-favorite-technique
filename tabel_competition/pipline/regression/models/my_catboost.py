import numpy as np
import pandas as pd
from catboost import CatBoost
from catboost import Pool
from sklearn.metrics import mean_squared_error


EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10000
CTB_NUM_ROUND = 10000

ctb_params = {
    "loss_function": "RMSE",
    "num_boost_round": CTB_NUM_ROUND,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}
# ctb_best_params = {
#     "learning_rate": 0.01,
#     "l2_leaf_reg": 1,
#     "depth": 9,
#     "bagging_temperature": 2,
#     "random_strength": 4
# }
ctb_best_params = {}
ctb_params.update(ctb_best_params)


def catboost_train(X_train, y_train, X_valid, y_valid):
    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)
    ctb_model = CatBoost(ctb_params)
    ctb_model.fit(train_pool,
                  eval_set=[valid_pool],
                  use_best_model=True,
                  verbose=500)
    ctbm_va_pred = ctb_model.predict(X_valid)
    ctb_valid_score = np.sqrt(mean_squared_error(y_valid, ctbm_va_pred))
    return ctb_model, ctb_valid_score
