# required library
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode(data, cat_features):
    le = LabelEncoder()
    for column in cat_features:
        le.fit(data[column])
        label_encoded_column = le.transform(data[column])
        data[column] = pd.Series(label_encoded_column).astype('category')
    return data
