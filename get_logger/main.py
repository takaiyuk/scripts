import os
import gc
import logging
from get_logger import *

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

VERSION = "9999"

def RMSE(true, pred):
    return np.round(mean_squared_error(true, pred)**0.5, 5)

lgb_params = {
    "objective" : "regression",
    "metric" : "rmse",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 15,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.7,
    "feature_fraction" : 0.7,
    "min_data_in_leaf": 80,
    "bagging_seed" : 42,
    "silent": -1,
    "seed": 42
}

def load_data():
    boston = load_boston()
    boston.keys()
    X = boston['data']
    y = boston['target']
    logger.info('X.shape: {}, y.shape: {}'.format(X.shape, y.shape))
    return X, y

def main():
    logger.info('lgb_params: {}'.format(lgb_params))
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info('X_train.shape: {}, X_test.shape: {}'.format(X_train.shape, X_val.shape))
    clf = lgb.LGBMRegressor(**lgb_params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm_logger(logger, period=10)] 
    )
    pred = clf.predict(X_val)
    score = RMSE(y_val, pred)
    logger.info('cv score: {}'.format(score))

if __name__ == '__main__':
    gc.enable()
    logger = get_logger(VERSION, path_prefix='logs')
    logger.info('VERSION: {}'.format(VERSION))
    main()
