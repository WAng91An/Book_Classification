import numpy as np
import pandas as pd
import json
import os
from src.utils import config
from src.utils.util import *
from src.embedding.embedding import Embedding

class MLData(object):
    """
    获取用于 ML 模型训练的数据：X_train，X_test，y_train，y_test
    """
    def __init__(self, debug_mode=False, train_mode=True):

        # 加载embedding， 如果不训练， 则不处理数据
        self.debug_mode = debug_mode
        self.em = Embedding()
        self.tfidf, self.w2v, self.fast, self.lda = self.em.load_model()
        if train_mode:
            self.preprocessor()

    def preprocessor(self):

        print('load data')
        self.train = pd.read_csv(config.train_data_path, sep='\t').dropna() # dropna 删除缺失数据
        self.dev = pd.read_csv(config.dev_data_path, sep='\t').dropna()

        if self.debug_mode:
            self.train = self.train.sample(n=1000).reset_index(drop=True) # reset_index：重置索引。不想保留原来的index，使用参数 drop=True，默认 False。
            self.dev = self.dev.sample(n=100).reset_index(drop=True)

        # 拼接数据
        self.train["text"] = self.train['title'] + self.train['desc']
        self.dev["text"] = self.dev['title'] + self.dev['desc']

        # 分词
        self.train["queryCut"] = self.train["text"].apply(query_cut)
        self.dev["queryCut"] = self.dev["text"].apply(query_cut)

        # 过滤停止词
        self.train["queryCutRMStopWord"] = self.train["queryCut"].apply(lambda x: [word for word in x if word not in get_stop_word_list()])
        self.dev["queryCutRMStopWord"] = self.dev["queryCut"].apply(lambda x: [word for word in x if word not in get_stop_word_list()])

        # 生成 label 与 id 的对应关系， 并保存到文件中， 如果存在这个文件则直接加载
        if os.path.exists(config.root_path + '/data/label2id.json'):
            labelNameToIndex = json.load(open(config.root_path + '/data/label2id.json', encoding='utf-8'))
        else:
            labelName = self.train['label'].unique()  # 全部label列表
            labelIndex = list(range(len(labelName)))  # 全部label标签
            labelNameToIndex = dict(zip(labelName, labelIndex))  # label的名字对应标签的字典
            with open(config.root_path + '/data/label2id.json', 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in labelNameToIndex.items()}, f)

        # map：https://blog.csdn.net/bqw18744018044/article/details/79963829
        self.train["labelIndex"] = self.train['label'].map(labelNameToIndex)
        self.dev["labelIndex"] = self.dev['label'].map(labelNameToIndex)

    def process_data(self, method='word2vec'):

        # 处理数据， 获取到数据的 embedding， 如tfidf ,word2vec, fasttext
        X_train = self.get_feature(self.train, method)
        X_test = self.get_feature(self.dev, method)
        y_train = self.train["labelIndex"]
        y_test = self.dev["labelIndex"]
        return X_train, X_test, y_train, y_test

    def get_feature(self, data, method='word2vec'):

        if method == 'tfidf':
            data = [' '.join(query) for query in data["queryCutRMStopWord"]]
            return self.tfidf.transform(data)
        elif method == 'word2vec':
            return np.vstack(data['queryCutRMStopWord'].apply(lambda x: sentence_to_vector(x, self.w2v)[0]))
        elif method == 'fasttext':
            return np.vstack(data['queryCutRMStopWord'].apply(lambda x: sentence_to_vector(x, self.fast)[0]))
        else:
            NotImplementedError
