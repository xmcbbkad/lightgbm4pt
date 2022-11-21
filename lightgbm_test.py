# coding: utf-8

import os
import re
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("loading data...")


class Lightgbm4pt():
    def __init__(self, feature_file):
        super(Lightgbm4pt, self).__init__()
        self.feature_names = self.read_feature_list(feature_file=feature_file)
        
        #https://lightgbm.apachecn.org/#/docs/6
        self.params = {
            "task" : "train",
            "boosting_type" : "gbdt",
            "objective" : "regression",
            "metric" : "l2",
            "max_bin" : 255,
            "num_iterations" : 100,
            "learning_rate" : 0.02,
            "num_leaves" : 31,
            "min_data_in_leaf" : 10,
            "feature_fraction" : 1,
            "bagging_fraction": 1,
        }

    def read_feature_list(self, feature_file='./features.txt', mode='use'):
        feature_list = []
        with open(feature_file, 'r') as f:
            for line in f.readlines():  
                line = line.strip()
                feature, use = line.split(' ')
                if (mode == 'use' and use == '1') or mode == 'all':
                    feature_list.append(feature)
        return feature_list        


    def load_data_from_file(self, data_file):
        df = pd.read_csv(data_file, header=0, sep='\t')
        df = df[self.feature_names]
    
        X = df.drop('t0_close_ratio_last', axis=1).values
        Y = df['t0_close_ratio_last'].values
    
        return X, Y


    def load_data_from_dir(self, data_dir, pattern='.', not_pattern='fffff'):
        df_all = pd.DataFrame(columns = self.feature_names)
        for filename in sorted(os.listdir(data_dir)):
            if (not re.search(pattern, filename)) or re.search(not_pattern, filename):
                #logger.info("filter_read_file={}, pattern={}, not_pattern={}".format(filename, pattern, not_pattern))
                continue
            #logger.info("read_file={}".format(filename))
            df = pd.read_csv(os.path.join(data_dir, filename), header=0, sep='\t')
            df = df[self.feature_names]
            df_all = df_all.append(df, ignore_index=True)
       
        X = df_all.drop('t0_close_ratio_last', axis=1).values
        Y = df_all['t0_close_ratio_last'].values
    
        return X, Y


    def train(self, x_train, y_train, x_eval, y_eval):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_eval, y_eval)

        
        #callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=30)]
        #callbacks = [lgb.early_stopping(stopping_rounds=10)]
        callbacks=[]
        self.gbm = lgb.train(self.params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        callbacks=callbacks)

        logger.info("saving model...")
        self.gbm.save_model('model.txt')

    def predict(self, test_data):
        y_pred = self.gbm.predict(test_data, num_iteration=self.gbm.best_iteration)
        return y_pred


    def cal_metrics(self, y_test, y_pred):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
       
        gain = 0
        long_gain = 0
        short_gain = 0
        
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        for i in range(len(y_test)):
            #logger.info("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
            #print("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
            if y_pred[i] > 1 and y_test[i] > 1:
                tp +=1
                long_gain += y_test[i]-1
            elif y_pred[i] > 1 and y_test[i] < 1:
                fp +=1
                long_gain -= (1-y_test[i])
            elif y_pred[i] < 1 and y_test[i] > 1:
                fn +=1
                short_gain -= (y_test[i]-1)
            elif y_pred[i] < 1 and y_test[i] < 1:
                tn +=1
                short_gain += (1-y_test[i])
       
        gain = round(long_gain + short_gain, 4)
        long_gain = round(long_gain, 4)
        short_gain = round(short_gain, 4)

        acc = round((tp+tn)*1.0/(tp+tn+fp+fn), 4)
        long_acc = round(tp*1.0/(tp+fp), 4)
        short_acc = round(tn*1.0/(tn+fn), 4)
        #logger.info("tp={}, fp={}, fn={}, tn={}, acc={}, long_acc={}, short_acc={}, gain={}, long_gain={}, short_gain={}".format(tp, fp, fn, tn ,acc, long_acc, short_acc, gain, long_gain, short_gain))
        m = {"rmse":rmse, "tp":tp, "fp":fp, "fn":fn, "tn":tn, "acc":acc, "long_acc":long_acc, "short_acc":short_acc, "gain":gain, "long_gain":long_gain, "short_gain":short_gain}
        logger.info("rmse={}, tp={}, fp={}, fn={}, tn={}, acc={}, long_acc={}, short_acc={}, gain={}, long_gain={}, short_gain={}".format(round(m['rmse'],4), m['tp'], m['fp'], m['fn'], m['tn'] ,format(m['acc'],'.2%'), format(m['long_acc'],'.2%'), format(m['short_acc'],'.2%'), format(m['gain'],'.2%'), format(m['long_gain'],'.2%'), format(m['short_gain'],'.2%')))
        return m

    def feature_importance(self, ):
        logger.info(pd.DataFrame({
                'column': self.feature_names[1:],
                'importance': self.gbm.feature_importance(),
            }).sort_values(by='importance', ascending=False))



    def split_train_eval_by_month(self, data_dir):
        month = []
        for filename in sorted(os.listdir(data_dir)):
            month.append(filename[:7])
        month = sorted(list(set(month)))

        for i in range(len(month)):
            logger.info('train:')
            x_train, y_train = self.load_data_from_dir(data_dir=data_dir, pattern='.', not_pattern=month[i])
            logger.info('eval:')
            x_eval, y_eval = self.load_data_from_dir(data_dir=data_dir, pattern=month[i], not_pattern='ffff')
            logger.info('------------------------------')

            lightgbm4pt.train(x_train, y_train, x_eval, y_eval)
            y_pred = lightgbm4pt.predict(x_eval)

            logger.info("eval_month={}".format(month[i]))
            lightgbm4pt.cal_metrics(y_eval, y_pred)
    
            y_random = np.random.rand(len(y_pred))
            y_random += 0.5
            logger.info('random_metrics:')
            m = lightgbm4pt.cal_metrics(y_eval, y_random)
       
    


if __name__ == '__main__':
    lightgbm4pt = Lightgbm4pt(feature_file = '/Users/xiaokunfan/code/candle_features/features.txt')
    lightgbm4pt.split_train_eval_by_month(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
    exit(0)



    x_train, y_train = lightgbm4pt.load_data_from_dir(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
    x_eval, y_eval  = lightgbm4pt.load_data_from_dir(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')

    lightgbm4pt.train(x_train, y_train, x_eval, y_eval)
    y_pred = lightgbm4pt.predict(x_eval)

    lightgbm4pt.cal_metrics(y_eval, y_pred)
    
    y_random = np.random.rand(len(y_pred))
    y_random += 0.5
    logger.info('random_metrics:')
    m = lightgbm4pt.cal_metrics(y_eval, y_random)


    lightgbm4pt.feature_importance()

    #--------------------------------------------------------------------------------

    pred_dir = '/Users/xiaokunfan/code/data/TSLA_5mins_features_202211'
    for filename in sorted(os.listdir(pred_dir)):
        x, y = lightgbm4pt.load_data_from_file(os.path.join(pred_dir, filename))
        y_pred = lightgbm4pt.predict(x)
        logger.info(filename)
        lightgbm4pt.cal_metrics(y, y_pred)
        lightgbm4pt.cal_metrics(y, y_pred)

        logger.info('random_metrics:')
        y_random = np.random.rand(len(y_pred))
        y_random += 0.5
        lightgbm4pt.cal_metrics(y, y_random)

        logger.info('--------------------------------------------------')

    x, y = lightgbm4pt.load_data_from_dir(data_dir=pred_dir)
    y_pred = lightgbm4pt.predict(x)
    lightgbm4pt.cal_metrics(y, y_pred)
    
    logger.info('random_metrics:')
    y_random = np.random.rand(len(y_pred))
    y_random += 0.5
    lightgbm4pt.cal_metrics(y, y_random)


