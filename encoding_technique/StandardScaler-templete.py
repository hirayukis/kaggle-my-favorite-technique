# required library
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# prepare data
X = pd.DataFrame()  # train data
test = pd.DataFrame()
num_features = []

# parameters
None

# main code
scalar = StandardScaler()
scalar.fit(X[num_features])
for df in [X, test]:
    df[num_features] = scalar.transform(df[num_features])
