# coding: utf-8

import os
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("loading data...")
'''
df_train = pd.read_csv('/Users/xiaokunfan/code/candle_features/2022-08-30_TSLA_tiger_5mins_features.csv', sep='\t')
df_test = pd.read_csv('/Users/xiaokunfan/code/candle_features/2022-08-30_TSLA_tiger_5mins_features.csv', sep='\t')

x_train = df_train['t0_close'].values
y_test = df_test['t0_close'].values
x_train = df_train.drop('t0_close', axis=1).values
y_test = df_test.drop('t0_close', axis=1).values

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
'''

def read_feature_list(feature_file='./features.txt', mode='use'):
    feature_list = []
    with open(feature_file, 'r') as f:
        for line in f.readlines():  
            line = line.strip()
            feature, use = line.split(' ')
            if (mode == 'use' and use == '1') or mode == 'all':
                feature_list.append(feature)
    return feature_list        

def load_data(data_dir):
    features = read_feature_list(feature_file='/Users/xiaokunfan/code/candle_features/features.txt', mode='use')
    df_all = pd.DataFrame(columns=features)
    for filename in os.listdir(data_dir):
        logger.info("read_file={}".format(filename))
        df = pd.read_csv(os.path.join(data_dir, filename), header=0, sep='\t')
        df = df[features]
        df_all = df_all.append(df, ignore_index=True)
   
    X = df_all.drop('t0_close_ratio_last', axis=1).values
    Y = df_all['t0_close_ratio_last'].values

    return features, X, Y

feature_names, x_train, y_train = load_data(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
lgb_train = lgb.Dataset(x_train, y_train)

feature_names, x_test, y_test  = load_data(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
lgb_eval = lgb.Dataset(x_test, y_test)


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
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

tp = 0
fp = 0
tn = 0
fn = 0

logger.info('The rmse of prediction is:{}'.format(mean_squared_error(y_test, y_pred) ** 0.5))
for i in range(len(y_test)):
    logger.info("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
    print("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
    if y_pred[i] > 1 and y_test[i] > 1:
        tp +=1
    elif y_pred[i] > 1 and y_test[i] < 1:
        fp +=1
    elif y_pred[i] < 1 and y_test[i] > 1:
        fn +=1
    elif y_pred[i] < 1 and y_test[i] < 1:
        tn +=1

acc = (tp+tn)*1.0/(tp+tn+fp+fn)

logger.info("tp={}, fp={}, fn={}, tn={}, acc={}".format(tp, fp, fn, tn ,acc))


logger.info(pd.DataFrame({
        'column': feature_names[1:],
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance', ascending=False))
#logger.info('Feature importances:{}'.format(list(gbm.feature_importance())))
