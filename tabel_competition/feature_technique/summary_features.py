# required library
import numpy as np
import pandas as pd

# prepare data
train = pd.DataFrame()
cat_features = []

# parameters
None

# create feautures
for cat in cat_features:
    train[f"{cat}_mean"] = train[[cat, "y"]].groupby(cat).mean()
    train[f"{cat}_max"] = train[[cat, "y"]].groupby(cat).max()
    train[f"{cat}_min"] = train[[cat, "y"]].groupby(cat).min()
    train[f"{cat}_std"] = train[[cat, "y"]].groupby(cat).std()
    train[f"{cat}_count"] = train[[cat, "y"]].groupby(cat).count()
    train[f"{cat}_quantile0.1"] = train[[cat, "y"]].groupby(cat).quantile(0.1)
    train[f"{cat}_quantile0.25"] = train[[cat, "y"]].groupby(cat).quantile(0.25)
    train[f"{cat}_quantile0.5"] = train[[cat, "y"]].groupby(cat).quantile(0.5)
    train[f"{cat}_quantile0.75"] = train[[cat, "y"]].groupby(cat).quantile(0.75)
    train[f"{cat}_quantile0.9"] = train[[cat, "y"]].groupby(cat).quantile(0.9)
