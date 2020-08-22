import numpy as np
import pandas as pd

from lib.clean_data import clean
from lib.create_features import create_features
from lib.encode_feature import label_encode
from models.my_lightgbm import lightgbm_train
from sklearn.model_selection import KFold

# fhase1. set parameters
TRAIN_DATA_PATH = "../../sample_data/regression_sample_train.csv"
TEST_DATA_PATH = "../../sample_data/regression_sample_test.csv"
cat_features = ["area"]
num_features = [
    "id", "position", "age", "sex", "partner", "num_child",
    "education", "service_length", "study_time", "commute", "overtime"
]
cat_encoding_method = "LabelEncoder"
target_col = "salary"
fold_num = 4
IS_MODEL_RUN = {
    "LightGBM": True
}
submission_path = "submission/"

# fhase1.5 check parameters
assert sum(IS_MODEL_RUN.values()) != 0

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

# fahse8. training
skf = KFold(n_splits=fold_num)
if IS_MODEL_RUN["LightGBM"]:
    lgb_pred_cv = np.zeros(len(test.index))

for i, indexs in enumerate(skf.split(X, y)):
    print(f"=====Fold: {i+1}=====")
    train_index, test_index = indexs
    X_train, y_train = X[train_index], y[train_index]
    X_valid, y_valid = X[test_index], y[test_index]
    print(f"X_train's shape: {X_train.shape}, X_test's shape: {X_test.shape}")
    # singel LightGBM
    if IS_MODEL_RUN["LightGBM"]:
        lgb_model, valid_score, importance_df = lightgbm_train(X_train, y_train, X_valid, y_valid, cols)
        lgb_submission = lgb_model.predict((X_test), num_iteration=lgb_model.best_iteration)
        lgb_pred_cv += lgb_submission / fold_num
        score = np.mean(valid_score)
        light_submission_df = pd.DataFrame(lgb_pred_cv)
        light_submission_df.columns = [target_col]
        light_submission_df.to_csv(submission_path + "submission_single_lgb.csv", index=False)
