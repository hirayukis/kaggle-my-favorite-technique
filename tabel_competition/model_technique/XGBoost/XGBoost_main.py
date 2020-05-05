# required library
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# prepare data
train = pd.DataFrame()
X = pd.DataFrame()  # train data without objective variable
test = pd.DataFrame()  # test data
y = train["y"]
id_name = "id"
object_var_name = "y"

# parameters
CV_FOLD_NUM = 4
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

# XGBoost main train and predict
params.update(tuning_params)
kf = KFold(n_splits=CV_FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []

pred_cv = np.zeros(len(test.index))

for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(params, xgb_dataset,
                     XGB_ROUND_NUM,
                     evals=[(xgb_dataset, 'train'), (xgb_test_dataset, 'eval')],
                     early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                     verbose_eval=VERBOSE_EVAL)
    xgbm_va_pred = xgbm.predict(xgb.DMatrix(X_valid))
    # 評価方法については調整
    score_ = np.sqrt(mean_squared_error(y_valid, xgbm_va_pred))
    scores.append(score_)

    xgbm_submission = xgbm.predict((test), num_iteration=xgbm.best_iteration)
    pred_cv += xgbm_submission / CV_FOLD_NUM


score = np.mean(scores)

xgbm_submission_df = pd.concat([test[[id_name]], pd.DataFrame(pred_cv)], axis=1)
xgbm_submission_df.columns = [id_name, object_var_name]
xgbm_submission_df.to_csv("submission_xgb.csv", index=False)
print("end")
