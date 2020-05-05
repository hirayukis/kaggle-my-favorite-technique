# required library
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
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

# LightGBM main train  and predict
params.update(tuning_params)
kf = KFold(n_splits=CV_FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
feature_importance_df = pd.DataFrame()
pred_cv = np.zeros(len(test.index))

for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    lgb_model = lgb.train(params, lgb_train,
                          num_boost_round=LGB_ROUND_NUM,
                          valid_names=["train", "valid"],
                          valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose_eval=VERBOSE_EVAL)
    if i == 0:
        importance_df = pd.DataFrame(lgb_model.feature_importance(), index=X.columns, columns=['importance'])
    else:
        importance_df += pd.DataFrame(lgb_model.feature_importance(), index=X.columns, columns=['importance'])
    va_pred = lgb_model.predict(X_valid)
    # 評価方法については調整
    score_ = np.sqrt(mean_squared_error(y_valid, va_pred))
    scores.append(score_)

    lgb_submission = lgb_model.predict((test), num_iteration=lgb_model.best_iteration)
    pred_cv += lgb_submission / CV_FOLD_NUM

score = np.mean(scores)

light_submission_df = pd.concat([test[[id_name]], pd.DataFrame(pred_cv)], axis=1)
light_submission_df.columns = [id_name, object_var_name]
light_submission_df.to_csv("submission_lgb.csv", index=False)
print("end")
