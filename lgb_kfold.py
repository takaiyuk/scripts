import numpy as np; np.random.random(42)
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings; warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.figsize']=(8,5)
import seaborn as sns; sns.set()


# https://www.kaggle.com/artgor/santander-eda-fe-fs-and-models
lgb_params = {'num_leaves': 63,
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

def RMSE(true, pred):
    return mean_squared_error(true, pred)**0.5

def display_importances(feature_importance_df_, score, save=True):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    if save:
        plt.savefig('lgb_importances_{}.png'.format(score))

def kfold_lgb(train, test, target, features, params, objective, es=100, n_folds=10):
    assert objective=="regression" or objective=="classification", print("objective must be regression or classification")
    print("objective is {}\n".format(objective))
    
    import time
    t0 = time.time()
        
    kfold = StratifiedKFold(n_folds, random_state=42)
    oof = np.zeros((train.shape[0]))
    sub = np.zeros((test.shape[0]))
    score = [0.0 for i in range(n_folds)]
    feature_importance_df = pd.DataFrame()

    for fold_, (tr_idx, val_idx) in enumerate(kfold.split(train[features], target)):
        X_train, X_val = train.iloc[tr_idx, :][features].values, train.iloc[val_idx, :][features].values
        y_train, y_val = target.iloc[tr_idx].values, target.iloc[val_idx].values
        X_test = test[features].values
        
        if objective=="regression":
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      eval_metric='rmse', early_stopping_rounds=es, verbose=1000,
                      callbacks=[lgbm_logger(logger, period=10)] )
            oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
            sub += model.predict(X_test, num_iteration=model.best_iteration_) / n_folds
            score[fold_] = RMSE(y_val, oof[val_idx])
            
        else:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 
                      eval_metric='auc', early_stopping_rounds=es, verbose=1000)
            oof[val_idx] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
            sub += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1] / n_folds
            score[fold_] = roc_auc_score(y_val, oof[val_idx])
            
        # Feature Importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print("Fold {}: {} (FOLD RUNTIME: {}[min.])\n".format( fold_+1, round(score[fold_], 5), round((time.time() - t0)/60,1) ))

    print( "\nscore:{} (std:{})".format(round(np.mean(score),5), round(np.std(score),5)) )
#     logger.info("score:{} (std:{})".format(round(np.mean(score),5), round(np.std(score),5)))
    
    return model, oof, sub, score, feature_importance_df


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
    target = train['target']
    features = train.drop(['target', 'ID'], axis=1).columns
    objective = "classification"
    es = 200
    n_folds = 10
    
    model, oof, sub, score, feature_importance_df = kfold_lgb(train, test, target, features, params=lgb_params, 
                                                              objective=objective, es=es, n_folds=n_folds)

    display_importances(feature_importance_df, score, save=True)
