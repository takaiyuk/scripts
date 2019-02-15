import numpy as np; np.random.random(42)
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# https://www.kaggle.com/artgor/santander-eda-fe-fs-and-models
params = {'num_leaves': 63,
          'n_estimators': 10000,
          'min_child_weight': 40, 
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.1,
          'boosting': 'gbdt',
          'feature_fraction': 0.9,
          'bagging_freq': 1,
          'bagging_fraction': 0.9,
          'bagging_seed': 11,
          'lambda_l1': 0.2,
          'lambda_l2': 0.2,
          'random_state': 42,
          'metric': 'auc',
          'verbosity': -1}

N_FOLDS = 5
features = train.columns.tolist()
target = train["target"].values
train.drop(["target"], axis=1, inplace=True)


folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(train_selected))
sub = np.zeros(len(test_selected))
feature_importance_df = pd.DataFrame()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_selected.values, target)):
    X_train, y_train = train.iloc[trn_idx][features], target[trn_idx]
    X_val, y_val = train.iloc[val_idx][features], target[val_idx]
    X_test = test.values
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)
    oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    sub += clf.predict_proba(X_test)[:, 1] / folds.n_splits
    score[fold_] = roc_auc_score(target[val_idx], oof[val_idx])
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = N_FOLDS + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print("Fold {}: {}".format(fold_+1, round(score[fold_],5)))

print("CV score(auc): {:<8.5f}, (std: {:<8.5f})".format(roc_auc_score(target, oof), np.std(score)))


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgb_importances.png')

display_importances(feature_importance_df)
