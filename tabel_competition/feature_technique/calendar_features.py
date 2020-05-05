# required library
import numpy as np
import pandas as pd

# prepare data
train = pd.DataFrame()

# parameters
date_col = ""

# create feautures
# とりあえずutcにしておくと、後々datetime型同士で加減法が使える
train[date_col] = pd.to_datetime(train[date_col], utc=True)
train[f"{date_col}_year"] = train[date_col].apply(lambda x: x.year)
train[f"{date_col}_month"] = train[date_col].apply(lambda x: x.month)
train[f"{date_col}_day"] = train[date_col].apply(lambda x: x.day)
train[f"{date_col}_hour"] = train[date_col].apply(lambda x: x.hour)
train[f"{date_col}_minute"] = train[date_col].apply(lambda x: x.minute)
train[f"{date_col}_second"] = train[date_col].apply(lambda x: x.second)
train[f"{date_col}_dayofweek"] = train[date_col].apply(lambda x: x.dayofweek)
