N_FOLDS = 5
features = train_selected.columns.tolist()

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
