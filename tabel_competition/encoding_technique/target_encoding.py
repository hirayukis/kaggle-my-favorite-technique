# required library
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# prepare data
X = pd.DataFrame()  # train data
target = pd.Series()  # target variable y["target"]
test = pd.DataFrame()
cat_features = []

# parameters
None

# main code
for c in cat_features:
    # 学習データ全体で、各カテゴリの置けるtargetの平均を計算
    data_tmp = pd.DataFrame({c: X[c], "target": target})
    target_mean = data_tmp.groupby(c)["target"].mean()
    # テストデータのカテゴリを置換
    test[c] = test[c].map(target_mean).astype(np.float)

    # 学習データの変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, X.shape[0])

    # 学習データを分割
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for idx_1, idx_2 in kf.split(X):
        target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()
        tmp[idx_2] = X[c].iloc[idx_2].map(target_mean)
    X[c] = tmp
