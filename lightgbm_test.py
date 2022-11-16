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


    def load_data_from_dir(self, data_dir):
        df_all = pd.DataFrame(columns = self.feature_names)
        for filename in os.listdir(data_dir):
            logger.info("read_file={}".format(filename))
            df = pd.read_csv(os.path.join(data_dir, filename), header=0, sep='\t')
            df = df[self.feature_names]
            df_all = df_all.append(df, ignore_index=True)
       
        X = df_all.drop('t0_close_ratio_last', axis=1).values
        Y = df_all['t0_close_ratio_last'].values
    
        return X, Y


    def train(self, x_train, y_train, x_eval, y_eval):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_eval, y_eval)

        
        callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=30)]
        self.gbm = lgb.train(self.params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        callbacks=callbacks)

        logger.info("saving model...")
        self.gbm.save_model('model.txt')

    def predict(self, test_data):
        logger.info('predicting...')
        y_pred = self.gbm.predict(test_data, num_iteration=self.gbm.best_iteration)
        return y_pred


    def cal_metrics(self, y_test, y_pred):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        logger.info('The rmse of prediction is:{}'.format(mean_squared_error(y_test, y_pred) ** 0.5))
        for i in range(len(y_test)):
            #logger.info("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
            #print("y_pred:{}  y_test:{}".format(round(y_pred[i], 4), y_test[i]))
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

    def feature_importance(self, ):
        logger.info(pd.DataFrame({
                'column': self.feature_names[1:],
                'importance': self.gbm.feature_importance(),
            }).sort_values(by='importance', ascending=False))


if __name__ == '__main__':
    lightgbm4pt = Lightgbm4pt(feature_file = '/Users/xiaokunfan/code/candle_features/features.txt')

    x_train, y_train = lightgbm4pt.load_data_from_dir(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
    x_eval, y_eval  = lightgbm4pt.load_data_from_dir(data_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')

    lightgbm4pt.train(x_train, y_train, x_eval, y_eval)
    y_pred = lightgbm4pt.predict(x_eval)

    lightgbm4pt.cal_metrics(y_eval, y_pred)

    pred_dir = '/Users/xiaokunfan/code/data/TSLA_5mins_features'
    for filename in os.listdir(pred_dir):
        x, y = lightgbm4pt.load_data_from_file(os.path.join(pred_dir, filename))
        y_pred = lightgbm4pt.predict(x)
        logger.info(filename)
        lightgbm4pt.cal_metrics(y, y_pred)


