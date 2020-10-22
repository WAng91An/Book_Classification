import numpy as np
import copy
from src.utils import config
from src.utils.util import sentence_to_vector
import pandas as pd
import joblib
import json
from tqdm import tqdm
import string
from sklearn import metrics
import jieba.posseg as pseg
from PIL import Image
import torchvision.transforms as transforms
import jieba
import lightgbm as lgb
import matplotlib.pyplot as plt
import torch
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from tqdm import tqdm
from datetime import timedelta
import time

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def Grid_Train_model(model, Train_features, Test_features, Train_label,
                     Test_label):
    # 构建训练模型并训练及预测
    # 网格搜索
    parameters = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [1000, 2000],
        'subsample': [0.6, 0.75, 0.9],
        'colsample_bytree': [0.6, 0.75, 0.9],
        'reg_alpha': [5, 10],
        'reg_lambda': [10, 30, 50]
    }
    # 有了 gridsearch 我们便不需要fit函数
    gsearch = GridSearchCV(model,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=3,
                           verbose=True)
    gsearch.fit(Train_features, Train_label)
    # 输出最好的参数
    print("Best parameters set found on development set:{}".format(
        gsearch.best_params_))
    return gsearch


def bayes_parameter_opt_lgb(trn_data,
                            init_round=3,
                            opt_round=5,
                            n_folds=5,
                            random_seed=6,
                            n_estimators=10000,
                            learning_rate=0.05):
    # 定义搜索参数
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {
            'application':
            'multiclass',
            'num_iterations':
            n_estimators,
            'learning_rate':
            learning_rate,
            'early_stopping_round':
            100,
            'num_class':
            len([
                x.strip() for x in open(config.root_path +
                                        '/data/class.txt').readlines()
            ]),
            'metric':
            'multi_logloss'
        }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params,
                           trn_data,
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,
                           verbose_eval=200)
        return max(cv_result['multi_logloss-mean'])
        # range
    # 搜索参数
    lgbBO = BayesianOptimization(lgb_eval, {
        'num_leaves': (24, 45),
        'feature_fraction': (0.1, 0.9),
        'bagging_fraction': (0.8, 1),
        'max_depth': (5, 8.99),
        'lambda_l1': (0, 5),
        'lambda_l2': (0, 3),
        'min_split_gain': (0.001, 0.1),
        'min_child_weight': (5, 50)
    },
                                 random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    print(lgbBO.max)
    return lgbBO.max


bayes_cv_tuner = BayesSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 application='multiclass',
                                 n_jobs=-1,
                                 verbose=1),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (2, 500),
        'max_depth': (0, 500),
        'min_child_samples': (0, 200),
        'max_bin': (100, 100000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (10, 10000),
    },
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=2),
    n_iter=30,
    verbose=1,
    refit=True)


def get_score(Train_label, Test_label, Train_predict_label,
              Test_predict_label):
    # 输出模型的准确率， 精确率，召回率， f1_score
    return metrics.accuracy_score(
        Train_label, Train_predict_label), metrics.accuracy_score(
            Test_label, Test_predict_label), metrics.recall_score(
                Test_label, Test_predict_label,
                average='micro'), metrics.f1_score(Test_label,
                                                   Test_predict_label,
                                                   average='weighted')

def formate_data(train, train_tfidf):
    # 将数据拼接到一起
    # pd.concat([train[[...]], train_tfidf, ], axis=1)
    Data = pd.concat([
        train[[
            'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], train_tfidf
    ] + [
        pd.DataFrame(
            train[feature_type].tolist(),
            columns=[feature_type + str(x) for x in range(train[feature_type].iloc[0].shape[0])])
        for feature_type in [
            'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
            'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
            'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
            'resnext_embedding', 'wide_embedding', 'bert_embedding', 'lda'
        ]
    ], axis=1).fillna(0.0)
    return Data

# def formate_data(train, train_tfidf, train_ae):
#     # 将数据拼接到一起
#     Data = pd.concat([
#         train[[
#             'labelIndex', 'length', 'capitals', 'caps_vs_length',
#             'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
#             'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
#             'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
#             'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
#             'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
#             'mean_word_len', 'punct_percent'
#         ]], train_tfidf, train_ae
#     ] + [
#         pd.DataFrame(
#             train[i].tolist(),
#             columns=[i + str(x) for x in range(train[i].iloc[0].shape[0])])
#         for i in [
#             'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
#             'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
#             'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
#             'resnext_embedding', 'wide_embedding', 'bert_embedding', 'lda'
#         ]
#     ], axis=1).fillna(0.0)
#     return Data