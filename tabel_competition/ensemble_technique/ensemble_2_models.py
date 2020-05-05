# required library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# prepare data
va_pred1 = np.array()  # prediction of validation data
va_pred2 = np.array()  # same shape as va_pred2
y_valid = np.array()
test = pd.DataFrame()
model1: "model that have a method predict"
model2: "model that have a method predict"

# parameters
None

# ensemble 2
# ENS
scores = []
alphas = []

for alpha in np.linspace(0, 1, 101):
    y_pred = alpha * va_pred1 + (1 - alpha) * va_pred2
    score = np.sqrt(mean_squared_error(np.exp(y_valid), y_pred))
    scores.append(score)
    alphas.append(alpha)

scores = np.array(scores)
alphas = np.array(alphas)

best_alpha = alphas[np.argmin(scores)]

print('best_score=', scores.min())
print('best_alpha=', best_alpha)
plt.plot(alphas, scores)
plt.title('score for ensemble')
plt.xlabel('alpha')
plt.ylabel('score')

score_ = scores.min()
scores.append(score_)

submission1 = np.exp(model1.predict((test), num_iteration=model1.best_iteration))
submission1[submission1 < 0] = 0

submission2 = np.exp(model2.predict(test))
submission2[submission2 < 0] = 0

submission = best_alpha * submission1 + (1 - best_alpha) * submission2
