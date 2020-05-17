# required library
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

# prepare data
train = pd.DataFrame()
test = pd.DataFrame()
X, y = train.drop("y", axis=1), train["y"]
cat_features = []

# parameters
# LightGBM parameters
FOLD_NUM = 4
LGB_NUM_ROUND = 10000
XGB_NUM_ROUND = 10000
CTB_NUM_ROUND = 10000

LGB_EARLY_STOPPING_ROUNDS = 50
XGB_EARLY_STOPPING_ROUNDS = 50
CTB_EARLY_STOPPING_ROUNDS = 50

light_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
    "learning_rate": 0.01
}
lgb_best_params = {
    "learning_rate": 0.0067052307664560535,
    "lambda_l1": 3.513541345228579e-05,
    "lambda_l2": 0.0010053075729828291,
    "num_leaves": 93,
    "feature_fraction": 0.518017214876781,
    "bagging_fraction": 0.7540574605800988,
    "bagging_freq": 3,
    "min_child_samples": 5
}
light_params.update(lgb_best_params)
# XGBoost parameters
xgb_params = {
    "eta": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
    "tree_method": "hist"
}
xgb_best_params = {
    "eta": 0.010235487563042057,
    "lambda": 1,
    "max_depth": 9,
    "alpha": 0.3600042513251054,
    "subsample": 0.7633887458218562,
    "gamma": 0
}
xgb_params.update(xgb_best_params)
# CatBoost parameters
ctb_params = {
    "loss_function": "RMSE",
    "num_boost_round": CTB_NUM_ROUND,
    "early_stopping_rounds": CTB_EARLY_STOPPING_ROUNDS,
}
ctb_best_params = {
    "learning_rate": 0.04396336192477268,
    "l2_leaf_reg": 1,
    "depth": 9,
    "bagging_temperature": 2,
    "random_strength": 4
}
xgb_params.update(ctb_best_params)

# ensemble
kf = KFold(n_splits=FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
feature_importance_df = pd.DataFrame()

pred_cv = np.zeros(len(test.index))
pred_cv_ctb = np.zeros(len(test.index))

for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]

    # LGB
    print(f"\n######### LightGBM Fold : {i}##################")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)
    lgbm = lgb.train(light_params,
                     lgb_train,
                     num_boost_round=LGB_NUM_ROUND,
                     valid_names=["train", "valid"],
                     valid_sets=[lgb_train, lgb_valid],
                     early_stopping_rounds=LGB_EARLY_STOPPING_ROUNDS,
                     verbose_eval=500)
    lgbm_va_pred = np.exp(lgbm.predict(X_valid, num_iteration=lgbm.best_iteration))
    lgbm_rmsle_score = np.sqrt(mean_squared_log_error(np.exp(y_valid), lgbm_va_pred))
    print(f"XGBoost score is : {lgbm_rmsle_score}")

    # LightGBMのみfeature importanceを取得する
    if i == 0:
        importance_df = pd.DataFrame(lgbm.feature_importance(), index=X.columns, columns=["importance"])
        oof_df = pd.DataFrame(lgbm_va_pred, X_valid.index, columns=["predict"])
    else:
        importance_df += pd.DataFrame(lgbm.feature_importance(), index=X.columns, columns=["importance"])
        oof_df = pd.concat([oof_df, pd.DataFrame(lgbm_va_pred, X_valid.index, columns=["predict"])])

    # XGB
    print(f"\n######### XGBoost Fold : {i}##################")
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params,
                     xgb_dataset,
                     num_boost_round=XGB_NUM_ROUND,
                     evals=[(xgb_dataset, "train"), (xgb_test_dataset, "eval")],
                     early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                     verbose_eval=500)
    xgbm_va_pred = np.exp(xgbm.predict(xgb.DMatrix(X_valid)))
    xgbm_rmsle_score = np.sqrt(mean_squared_log_error(np.exp(y_valid), xgbm_va_pred))
    print(f"XGBoost score is : {xgbm_rmsle_score}")

    # CatBoost
    print(f"\n######### CatBoost Fold : {i}##################")
    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)
    ctbm = CatBoost(ctb_params)
    ctbm.fit(train_pool,
             eval_set=[valid_pool],
             use_best_model=True,
             verbose=500)
    ctbm_va_pred = np.exp((ctbm.predict(X_valid)))
    ctb_rmsle_score = np.sqrt(mean_squared_log_error(np.exp(y_valid), ctbm_va_pred))
    print(f"CatBoost score is : {ctb_rmsle_score}")

    # ENS
    print(f"\n######### Ensemble Fold : {i}##################")
    lgb_xgb_ctb_rmsle = []
    lgb_xgb_ctb_alphas_betas = []

    for alpha in np.linspace(0, 1, 101):
        for beta in np.linspace(0, 1, 101):
            y_pred = beta * (alpha * lgbm_va_pred + (1 - alpha) * xgbm_va_pred) + (1 - beta) * ctbm_va_pred
            rmsle_score = np.sqrt(mean_squared_log_error(np.exp(y_valid), y_pred))
            lgb_xgb_ctb_rmsle.append(rmsle_score)
            lgb_xgb_ctb_alphas_betas.append([alpha, beta])

    lgb_xgb_ctb_rmsle = np.array(lgb_xgb_ctb_rmsle)
    lgb_xgb_ctb_alphas_betas = np.array(lgb_xgb_ctb_alphas_betas)

    lgb_xgb_best_alpha = lgb_xgb_ctb_alphas_betas[np.argmin(lgb_xgb_ctb_rmsle)]

    print("best_rmsle=", lgb_xgb_ctb_rmsle.min())
    print("best [alpha, beta]: ", lgb_xgb_best_alpha)

    score_ = lgb_xgb_ctb_rmsle.min()
    scores.append(score_)

    lgb_submission = np.exp(lgbm.predict((test), num_iteration=lgbm.best_iteration))

    xgbm_submission = np.exp(xgbm.predict(xgb.DMatrix(test)))

    ctbm_submission = np.exp(ctbm.predict(test))

    best_alpha, best_beta = lgb_xgb_best_alpha
    submission = beta * (alpha * lgb_submission + (1 - alpha) * xgbm_submission) + (1 - beta) * ctbm_submission

    pred_cv += submission / FOLD_NUM

oof_df = pd.concat([oof_df, np.exp(y)], axis=1)
oof_df = pd.concat([oof_df, X], axis=1)
print("##########")
print(np.mean(scores))
