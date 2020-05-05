# required library
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd

# prepare data
X = pd.DataFrame()  # train data
lgb_model = lgb.train()

# parameters
None

# plot feature importance
attr = {k: v for k, v in zip(X.columns, lgb_model.feature_importance()) if v > 0}
attr = sorted(attr.items(), key=lambda x: x[1], reverse=False)
x1, y1 = zip(*attr)
i1 = range(len(x1))
plt.figure(num=None, figsize=(9, 7), dpi=100, facecolor='w', edgecolor='k')
plt.barh(i1, y1)
plt.title("LGBM importance")
plt.yticks(i1, x1)
plt.show()
