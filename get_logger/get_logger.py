"""
- logger についての使い方は olivier の kernel を参考にした
https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480/code

- lightgbm の学習履歴を callback を利用して logger に出力する方法は amaotone さんのブログを参考にした
https://amalog.hateblo.jp/entry/lightgbm-logging-callback

- スクリプトのバージョン指定は必須で、ログ出力用のディレクトリの指定はオプション

- print() の代わりに logger.info() を使用することでログ出力可能
"""

import os
import logging
import numpy as np
from lightgbm.callback import _format_eval_result

def make_new_version(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    logs = os.listdir(log_dir)
    logs = [int(log.replace(".log", "")) for log in logs if ".log" in log]
    
    if logs==[]:
        new_version = 0
    else:
        new_version = np.max(logs)+1
        
    if len(str(new_version))==1:
        new_version = "00" + str(new_version)
    elif len(str(new_version))==2:
        new_version = "0" + str(new_version)
    else:
        new_version = str(new_version)
    
    return new_version

def make_logs_dir(log_dir=None):
    if not log_dir==None:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

def get_logger(VERSION, path_prefix=None):
    make_logs_dir(log_dir=path_prefix)
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    if path_prefix==None:
        fh = logging.FileHandler('{}.log'.format(VERSION))
    else:
        fh = logging.FileHandler('{}/{}.log'.format(path_prefix, VERSION))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_

def lgbm_logger(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback
