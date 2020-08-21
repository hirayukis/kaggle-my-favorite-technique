import pandas as pd

from lib.clean_data import clean
from lib.feature_handler import create_features
from models.my_lightgbm import lightgbm_train
from sklearn.model_selection import StratifiedKFold

# fhase1. set parameters
TRAIN_DATA_PATH = "../../sample_data/regression_sample_train.csv"
TEST_DATA_PATH = "../../sample_data/regression_sample_test.csv"
cat_features = []
num_features = []
target_col = "salary"
fold_num = 4
IS_MODEL_RUN = {
    "LightGBM": True
}

# fhase2. read data and merge
train = pd.read_csv(TRAIN_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)
y = train[target_col]
data = pd.concat([train.drop(target_col, axis=1), test])
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")
print(f"data merged. This shape: {data.shape}")

# fhase3. clean data
data, num_features, cat_features = clean(data, num_features, cat_features)

# fhase4. create features
data, num_features, cat_features = create_features(data, num_features, cat_features)

# fhase5. recreate train test
train = data[:len(train.index)]
test = data[len(train.index):]

# fhase6. adversarial validation

# fahse7. training
# singel LightGBM
if IS_MODEL_RUN["LightGBM"]:
    lightgbm_train()
