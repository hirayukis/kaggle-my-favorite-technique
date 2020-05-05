# required library
import numpy as np
import pandas as pd

# prepare data
train = pd.DataFrame()
time_features = []
windows = [3, 5, 10, 50, 100, 500, 1000]

# parameters
None


# create feautures
def calc_roll_stats(s, feature_name, windows=windows):
    '''
    Calculates rolling stats like mean, std, min, max...
    '''
    roll_stats = pd.DataFrame()
    for w in windows:
        roll_stats[feature_name + '_roll_mean_' + str(w)] = s.rolling(window=w, min_periods=1).mean()
        roll_stats[feature_name + '_roll_std_' + str(w)] = s.rolling(window=w, min_periods=1).std()
        roll_stats[feature_name + '_roll_min_' + str(w)] = s.rolling(window=w, min_periods=1).min()
        roll_stats[feature_name + '_roll_max_' + str(w)] = s.rolling(window=w, min_periods=1).max()
        roll_stats[feature_name + '_roll_range_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_' + str(w)]

        # 未来のデfeature_name + ー_タに利用できる時のみ
        roll_stats[feature_name + '_roll_mean_s_' + str(w)] = s.rolling(window=w, min_periods=1).mean().shift(-w)
        roll_stats[feature_name + '_roll_std_s_' + str(w)] = s.rolling(window=w, min_periods=1).std().shift(-w)
        roll_stats[feature_name + '_roll_min_s_' + str(w)] = s.rolling(window=w, min_periods=1).min().shift(-w)
        roll_stats[feature_name + '_roll_max_s_' + str(w)] = s.rolling(window=w, min_periods=1).max().shift(-w)
        roll_stats[feature_name + '_roll_range_s_' + str(w)] = roll_stats['roll_max_s_' + str(w)] - roll_stats['roll_min_s_' + str(w)]

        roll_stats[feature_name + '_roll_q10_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.10).shift()
        roll_stats[feature_name + '_roll_q25_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.25).shift()
        roll_stats[feature_name + '_roll_q50_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.50).shift()
        roll_stats[feature_name + '_roll_q75_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.75).shift()
        roll_stats[feature_name + '_roll_q90_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.90).shift()

        roll_stats[feature_name + '_mean_abs_chg' + str(w)] = roll_stats.apply(lambda x: np.mean(np.abs(np.diff(x))))

    roll_stats = roll_stats.fillna(value=0)

    return roll_stats


for time_feature in time_features:
    target = train[time_feature]
    roll_stats = calc_roll_stats(target, time_feature, windows)
    train = pd.concat([train, roll_stats], axis=1)
    train[f'{time_feature}+1'] = [0, ] + list(train[time_feature].values[:-1])
    train[f'{time_feature}-1'] = list(train[time_feature].values[1:]) + [0]
