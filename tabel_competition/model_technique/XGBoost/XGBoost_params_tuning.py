# required library
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# prepare data
train = pd.DataFrame()
X = pd.DataFrame()  # train data without objective variable
y = train["y"]

# parameters
CV_FOLD_NUM = 4
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10000
XGB_ROUND_NUM = 10000

objective = 'reg:squarederror'
metric = 'rmse'
direction = 'minimize'
optuna_trial_num = 100


# optuna training

def objective(trial):
    params = {
        'objective': objective,
        'eval_metric': metric,
        'seed': 42,
        'tree_method': 'hist',
        "eta": trial.suggest_loguniform('eta', 0.01, 0.2),
        'lambda': trial.suggest_int('lambda', 0, 2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        "alpha": trial.suggest_uniform('alpha', 0, 1.0),
        "subsample": trial.suggest_loguniform('subsample', 0.5, 1.0),
        "gamma": trial.suggest_int('gamma', 0, 3),
    }
    kf = KFold(n_splits=CV_FOLD_NUM,
               shuffle=True,
               random_state=42)
    scores = []

    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]
        # XGB
        xgb_dataset = xgb.DMatrix(X_train, label=y_train)
        xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
        xgbm = xgb.train(params, xgb_dataset,
                         XGB_ROUND_NUM,
                         evals=[(xgb_dataset, 'train'), (xgb_test_dataset, 'eval')],
                         early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                         verbose_eval=VERBOSE_EVAL)
        xgbm_va_pred = xgbm.predict(xgb.DMatrix(X_valid))
        xgbm_va_pred[xgbm_va_pred < 0] = 0
        score_ = np.sqrt(mean_squared_error(y_valid, xgbm_va_pred))
        scores.append(score_)

    return np.mean(scores)


study = optuna.create_study(direction=direction)
study.optimize(objective, n_trials=optuna_trial_num)

# 結果の確認
print('Best trial:')
xgb_trial = study.best_trial

print('  Value: {}'.format(xgb_trial.value))

print('  Params: ')

with open("xgbmparams.txt", "w") as file:
    for key, value in xgb_trial.params.items():
        print('    "{}": {},'.format(key, value))
        file.write('"{}": {},'.format(key, value))
