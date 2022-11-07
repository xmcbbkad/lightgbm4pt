# coding: utf-8

import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("loading data...")
df_train = pd.read_csv('./rlt_output_train_20180903_alldata', header=None, sep=' ')
df_test = pd.read_csv('./rlt_output_test_20180903_alldata', header=None, sep=' ')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#https://lightgbm.apachecn.org/#/docs/6
params = {
    "task" : "train",
    "boosting_type" : "gbdt",
    "objective" : "regression",
    "metric" : "l2",
    "max_bin" : 255,
    "num_trees" : 100,
    "learning_rate" : 0.02,
    "num_leaves" : 31,
    "min_data_in_leaf" : 10,
    "feature_fraction" : 1,
    "bagging_fraction": 1,
}

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)

logger.info("saving model...")
gbm.save_model('model.txt')

logger.info('predicting...')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

logger.info('The rmse of prediction is:{}'.format(mean_squared_error(y_test, y_pred) ** 0.5))

logger.info('Feature importances:{}'.format(list(gbm.feature_importance())))
