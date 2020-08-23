import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error


EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10000
LGB_ROUND_NUM = 10000
objective = 'regression'
metric = 'rmse'
tuning_params = {"learning_rate": 0.001,
                 'lambda_l1': 0,
                 'lambda_l2': 0,
                 'num_leaves': 256,
                 'feature_fraction': 1.0,
                 'bagging_fraction': 1.0,
                 'bagging_freq': 1,
                 'min_child_samples': 100}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric': metric,
    'verbosity': -1,
    "seed": 42,
}


def plot_feature_importance(cols, lgb_model):
    attr = {k: v for k, v in zip(cols, lgb_model.feature_importance()) if v > 0}
    attr = sorted(attr.items(), key=lambda x: x[1], reverse=False)
    x1, y1 = zip(*attr)
    i1 = range(len(x1))
    plt.figure(num=None, figsize=(9, 7), dpi=100, facecolor='w', edgecolor='k')
    plt.barh(i1, y1)
    plt.title("LGBM importance")
    plt.yticks(i1, x1)
    plt.save("eda_image/lightgbm_importance.png")


def lightgbm_train(X_train, y_train, X_valid, y_valid, cols):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    lgb_model = lgb.train(params, lgb_train,
                          num_boost_round=LGB_ROUND_NUM,
                          valid_names=["train", "valid"],
                          valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose_eval=VERBOSE_EVAL)
    importance_df = pd.DataFrame(lgb_model.feature_importance(), index=cols, columns=['importance'])
    va_pred = lgb_model.predict(X_valid)
    valid_score = np.sqrt(mean_squared_error(y_valid, va_pred))
    return lgb_model, valid_score, importance_df
