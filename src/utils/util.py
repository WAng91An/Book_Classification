# -*- coding: utf-8 -*-
import csv
import logging
from logging import handlers
from src.utils import config
import jieba
import numpy

def get_corpus(path, tf_idf=False, w2v=False):
    data = csv.reader(open(path, encoding="utf-8"))
    data_list = []
    for n, text in enumerate(data):
        if n == 0:
            continue
        if tf_idf:
            data_list.append(text[0])
        elif w2v:
            data_list.append(text[0].split(" "))
        else:  # dl
            data_list.extend(text[0].split(" "))
    return data_list


def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger

def query_cut(query):
    return list(jieba.cut(query))


def wam(sentence, w2v_model, method='mean', aggregate=True):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    method： 聚合方法 mean 或者max
    aggregate: 是否进行聚合
    @return:
    '''
    arr = np.array([
        w2v_model.wv.get_vector(s) for s in sentence
        if s in w2v_model.wv.vocab.keys()
    ])
    if not aggregate:
        return arr
    if len(arr) > 0:
        # 第一种方法对一条样本中的词求平均
        if method == 'mean':
            return np.mean(np.array(arr), axis=0)
        # 第二种方法返回一条样本中的最大值
        elif method == 'max':
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)

def get_stop_word_list():

    data_stop_list = open(config.stop_words_path).readlines()
    data_stop_list = [i.strip() for i in data_stop_list]
    data_stop_list.append(" ")
    data_stop_list.append("\n")
    return data_stop_list