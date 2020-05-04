# required library
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# prepare data
X = pd.DataFrame()  # train data
test = pd.DataFrame()
cat_features = []

# parameters
None

# main code
for df in [X, test]:
    le = LabelEncoder()
    for column in cat_features:
        le.fit(df[column])
        label_encoded_column = le.transform(df[column])
        df[column] = pd.Series(label_encoded_column).astype('category')
