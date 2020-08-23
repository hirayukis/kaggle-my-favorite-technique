import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

from lib.clean_data import clean
from lib.create_features import create_features
from lib.encode_feature import label_encode
from models.my_lightgbm import lightgbm_train
from models.my_xgboost import xgboost_train
from models.my_catboost import catboost_train
from models.my_svm import svr_train
from models.my_knn import knnr_train
from models.my_ridge import ridge_train
from models.my_linear_regression import linear_regression_train
from models.my_random_forest import randomforest_train
from models.my_bagging_regressor import bagging_regressor_train
from sklearn.model_selection import KFold

# fhase1. set parameters
# file path
TRAIN_DATA_PATH = "../../sample_data/regression_sample_train.csv"
TEST_DATA_PATH = "../../sample_data/regression_sample_test.csv"
# features
cat_features = ["area"]
num_features = [
    "id", "position", "age", "sex", "partner", "num_child",
    "education", "service_length", "study_time", "commute", "overtime"
]
target_col = "salary"
# encoding method
cat_encoding_method = "LabelEncoder"
# kFold
fold_num = 4
# what to do in a single model
IS_MODEL_RUN = {
    "LightGBM": True,
    "XGBoost": True,
    "CatBoost": True,
    "SVR": True,
    "KNNR": True,
    "RandomForest": True,
    "Ridge": True,
    "LinearRegression": True,
    "BaggingRegressor": True,
}
# simple ensemble
do_ensemble = False
simple_ensemble = {
    "LightGBM": .8,
    "XGBoost": .1,
    "CatBoost": .0,
    "SVR": .1,
    "KNNR": 0,
    "RandomForest": 0,
    "Ridge": 0,
    "LinearRegression": 0,
    "BaggingRegressor": 0,
}
# save path
submission_path = "submission/"
# random seed
seed = 42

# fhase1.5 check parameters
assert sum(IS_MODEL_RUN.values()) != 0
assert simple_ensemble.keys() == IS_MODEL_RUN.keys()
if do_ensemble:
    for modelname in simple_ensemble.keys():
        if simple_ensemble[modelname] > 0:
            print(f"ensemble model: {modelname}")
            assert IS_MODEL_RUN[modelname]

# fhase2. read data and merge
train = pd.read_csv(TRAIN_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)
y = train[target_col].values
data = pd.concat([train.drop(target_col, axis=1), test])
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")
print(f"data merged. This shape: {data.shape}")

# fhase3. clean data
data, num_features, cat_features = clean(data, num_features, cat_features)

# fhase4. create features
data, num_features, cat_features = create_features(data, num_features, cat_features)

# fhase5. encoding
if cat_encoding_method == "LabelEncoder":
    data = label_encode(data, cat_features)

# fhase6. recreate train test
X = data[:len(train.index)].values
X_test = data[len(train.index):].values
cols = data.columns
print(f"recreate X shape: {X.shape}")
print(f"recreate test shape: {X_test.shape}")

# fhase7. adversarial validation

# fahse8. First sibgle training and referrence
kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
if IS_MODEL_RUN["LightGBM"]:
    lgb_pred_cv = np.zeros(len(test.index))
    lgb_valid_scores = []
if IS_MODEL_RUN["XGBoost"]:
    xgb_pred_cv = np.zeros(len(test.index))
    xgb_valid_scores = []
if IS_MODEL_RUN["CatBoost"]:
    cbt_pred_cv = np.zeros(len(test.index))
    cbt_valid_scores = []
if IS_MODEL_RUN["SVR"]:
    svr_pred_cv = np.zeros(len(test.index))
    svr_valid_scores = []
if IS_MODEL_RUN["KNNR"]:
    knnr_pred_cv = np.zeros(len(test.index))
    knnr_valid_scores = []
if IS_MODEL_RUN["RandomForest"]:
    rf_pred_cv = np.zeros(len(test.index))
    rf_valid_scores = []
if IS_MODEL_RUN["Ridge"]:
    ridge_pred_cv = np.zeros(len(test.index))
    ridge_valid_scores = []
if IS_MODEL_RUN["LinearRegression"]:
    lr_pred_cv = np.zeros(len(test.index))
    lr_valid_scores = []
if IS_MODEL_RUN["BaggingRegressor"]:
    br_pred_cv = np.zeros(len(test.index))
    br_valid_scores = []

for i, indexs in enumerate(kf.split(X, y)):
    print(f"\n=====Fold: {i+1}=====")
    train_index, test_index = indexs
    X_train, y_train = X[train_index], y[train_index]
    X_valid, y_valid = X[test_index], y[test_index]
    print(f"X_train's shape: {X_train.shape}, X_test's shape: {X_test.shape}")
    # singel LightGBM
    if IS_MODEL_RUN["LightGBM"]:
        lgb_model, lgb_valid_score, importance_df = lightgbm_train(X_train, y_train, X_valid, y_valid, cols)
        print(f"Fold {i+1} LightGBM valid score is: {lgb_valid_score}")
        lgb_valid_scores.append(lgb_valid_score)
        lgb_submission = lgb_model.predict((X_test), num_iteration=lgb_model.best_iteration)
        lgb_pred_cv += lgb_submission / fold_num
        light_submission_df = pd.DataFrame(lgb_pred_cv)
        light_submission_df.columns = [target_col]
        light_submission_df.to_csv(submission_path + "submission_single_lgb.csv", index=False)
    if IS_MODEL_RUN["XGBoost"]:
        xgb_model, xgb_valid_score = xgboost_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} XGBoost valid score is: {xgb_valid_score}")
        xgb_valid_scores.append(xgb_valid_score)
        xgb_submission = xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_iteration)
        xgb_pred_cv += xgb_submission / fold_num
        xgb_submission_df = pd.DataFrame(xgb_pred_cv)
        xgb_submission_df.columns = [target_col]
        xgb_submission_df.to_csv(submission_path + "submission_single_xgb.csv", index=False)
    if IS_MODEL_RUN["CatBoost"]:
        cbt_model, cbt_valid_score = catboost_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} CatBoost valid score is: {cbt_valid_score}")
        cbt_valid_scores.append(cbt_valid_score)
        cbt_submission = cbt_model.predict(X_test)
        cbt_pred_cv += cbt_submission / fold_num
        cbt_submission_df = pd.DataFrame(cbt_pred_cv)
        cbt_submission_df.columns = [target_col]
        cbt_submission_df.to_csv(submission_path + "submission_single_cbt.csv", index=False)
    if IS_MODEL_RUN["SVR"]:
        svr_model, svr_valid_score = svr_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} SVR valid score is: {svr_valid_score}")
        svr_valid_scores.append(svr_valid_score)
        svr_submission = svr_model.predict(X_test)
        svr_pred_cv += svr_submission / fold_num
        svr_submission_df = pd.DataFrame(svr_pred_cv)
        svr_submission_df.columns = [target_col]
        svr_submission_df.to_csv(submission_path + "submission_single_svr.csv", index=False)
    if IS_MODEL_RUN["KNNR"]:
        knnr_model, knnr_valid_score = knnr_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} KNN Regression valid score is: {knnr_valid_score}")
        knnr_valid_scores.append(knnr_valid_score)
        knnr_submission = knnr_model.predict(X_test)
        knnr_pred_cv += knnr_submission / fold_num
        knnr_submission_df = pd.DataFrame(knnr_pred_cv)
        knnr_submission_df.columns = [target_col]
        knnr_submission_df.to_csv(submission_path + "submission_single_knnr.csv", index=False)
    if IS_MODEL_RUN["RandomForest"]:
        rf_model, rf_valid_score = randomforest_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} RandomForest valid score is: {rf_valid_score}")
        rf_valid_scores.append(rf_valid_score)
        rf_submission = rf_model.predict(X_test)
        rf_pred_cv += rf_submission / fold_num
        rf_submission_df = pd.DataFrame(rf_pred_cv)
        rf_submission_df.columns = [target_col]
        rf_submission_df.to_csv(submission_path + "submission_single_rf.csv", index=False)
    if IS_MODEL_RUN["Ridge"]:
        ridge_model, ridge_valid_score = ridge_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} Ridge valid score is: {ridge_valid_score}")
        ridge_valid_scores.append(ridge_valid_score)
        ridge_submission = ridge_model.predict(X_test)
        ridge_pred_cv += ridge_submission / fold_num
        ridge_submission_df = pd.DataFrame(ridge_pred_cv)
        ridge_submission_df.columns = [target_col]
        ridge_submission_df.to_csv(submission_path + "submission_single_ridge.csv", index=False)
    if IS_MODEL_RUN["LinearRegression"]:
        lr_model, lr_valid_score = linear_regression_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} LinearRegression valid score is: {lr_valid_score}")
        lr_valid_scores.append(lr_valid_score)
        lr_submission = lr_model.predict(X_test)
        lr_pred_cv += lr_submission / fold_num
        lr_submission_df = pd.DataFrame(lr_pred_cv)
        lr_submission_df.columns = [target_col]
        lr_submission_df.to_csv(submission_path + "submission_single_lr.csv", index=False)
    if IS_MODEL_RUN["BaggingRegressor"]:
        br_model, br_valid_score = bagging_regressor_train(X_train, y_train, X_valid, y_valid)
        print(f"Fold {i+1} BaggingRegressor valid score is: {br_valid_score}")
        br_valid_scores.append(br_valid_score)
        br_submission = br_model.predict(X_test)
        br_pred_cv += br_submission / fold_num
        br_submission_df = pd.DataFrame(br_pred_cv)
        br_submission_df.columns = [target_col]
        br_submission_df.to_csv(submission_path + "submission_single_br.csv", index=False)


print("#" * 15 + "ALL SINGLE MODEL CV" + "#" * 15)
if IS_MODEL_RUN["LightGBM"]:
    print(f"LightGBM valid CV score is: {np.array(lgb_valid_scores).mean()}")
if IS_MODEL_RUN["XGBoost"]:
    print(f"XGBoost valid CV score is: {np.array(xgb_valid_scores).mean()}")
if IS_MODEL_RUN["CatBoost"]:
    print(f"Catboost valid CV score is: {np.array(cbt_valid_scores).mean()}")
if IS_MODEL_RUN["SVR"]:
    print(f"SVR valid CV score is: {np.array(svr_valid_scores).mean()}")
if IS_MODEL_RUN["KNNR"]:
    print(f"KNNR valid CV score is: {np.array(knnr_valid_scores).mean()}")
if IS_MODEL_RUN["RandomForest"]:
    print(f"RandomForest valid CV score is: {np.array(rf_valid_scores).mean()}")
if IS_MODEL_RUN["Ridge"]:
    print(f"Ridge valid CV score is: {np.array(ridge_valid_scores).mean()}")
if IS_MODEL_RUN["LinearRegression"]:
    print(f"LinearRegression valid CV score is: {np.array(lr_valid_scores).mean()}")
if IS_MODEL_RUN["BaggingRegressor"]:
    print(f"BaggingRegressor(SVR) valid CV score is: {np.array(br_valid_scores).mean()}")
print("#" * 15 + "ALL SINGLE MODEL CV" + "#" * 15)

# fhase9: simple ensemble
