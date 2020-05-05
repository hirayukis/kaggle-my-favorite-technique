# required library
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# prepare data
train = pd.DataFrame()
X = pd.DataFrame()  # train data without objective variable
lgb_model = lgb.train()
y = train["y"]

# parameters
CV_FOLD_NUM = 4
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10000
LGB_ROUND_NUM = 10000

objective = 'regression'
metric = 'rmse'
direction = 'minimize'
optuna_trial_num = 100


# optuna training
def objective(trial):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metric,
        'verbosity': -1,
        "seed": 42,
        "learning_rate": trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    kf = KFold(n_splits=CV_FOLD_NUM,
               shuffle=True,
               random_state=42)
    scores = []

    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        model = lgb.train(params, lgb_train, num_boost_round=LGB_ROUND_NUM,
                          valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose_eval=VERBOSE_EVAL)
        va_pred = model.predict(X_valid)
        score_ = np.sqrt(mean_squared_error(y_valid, va_pred))
        scores.append(score_)

    return np.mean(scores)


study = optuna.create_study(direction=direction)
study.optimize(objective, n_trials=optuna_trial_num)

# 結果の確認
print('Best trial:')
light_trial = study.best_trial

print('  Value: {}'.format(light_trial.value))

print('  Params: ')

with open("lightgbm_best_params.txt", "w") as file:
    for key, value in light_trial.params.items():
        print('    "{}": {},'.format(key, value))
        file.write('"{}": {},'.format(key, value))
