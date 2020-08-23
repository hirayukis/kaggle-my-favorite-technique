import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_feat(y_trues, y_preds, labels, est, filename, cols, x_max=1.0):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax[0].plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax[0].legend()
    ax[0].grid()
    ax[0].plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax[0].set_title('ROC curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_xlim([-0.01, x_max])
    _ = ax[0].set_ylabel('True Positive Rate')

    values = est.feature_importance()

    importance = pd.DataFrame(data=values, index=cols,
                              columns=['score']).sort_values(by='score',
                                                             ascending=False)

    sns.barplot(x=importance.score.iloc[:20],
                y=importance.index[:20],
                orient='h',
                palette='Reds_r', ax=ax[1])
    ax[1].set_title('Feature Importances')
    plt.savefig(filename + "_importance_feature.png")


def adversarial_validate(data, splitnum, filename=""):
    train = data[:splitnum]
    test = data[splitnum:]
    adv_train = train.copy()
    adv_test = test.copy()

    adv_train['dataset_label'] = 0
    adv_test['dataset_label'] = 1
    adv_master = pd.concat([adv_train, adv_test], axis=0)

    adv_X = adv_master.drop('dataset_label', axis=1)
    adv_y = adv_master['dataset_label']
    adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(adv_X,
                                                                        adv_y,
                                                                        test_size=0.4,
                                                                        stratify=adv_y,
                                                                        random_state=42)
    params = {
        'task': 'train',
        'objective': 'binary',
        'metric': 'binary_logloss',
        "seed": 42,
    }
    lgb_train = lgb.Dataset(adv_X_train, adv_y_train)
    lgb_valid = lgb.Dataset(adv_X_test, adv_y_test)
    lgb_model = lgb.train(params, lgb_train,
                          num_boost_round=10000,
                          valid_names=["train", "valid"],
                          valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=10,
                          verbose_eval=-1)
    validation = lgb_model.predict(adv_X_test)
    plot_roc_feat(
        [adv_y_test],
        [validation],
        ['Baseline'],
        lgb_model,
        filename,
        data.columns
    )
