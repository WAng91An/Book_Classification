# -*- coding: utf-8 -*-
import os
import re
import torch

class Const():
    def __init__(self):
        self.root_path = os.path.split(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])[0]
        self.model_path = os.path.join(self.root_path, "Book_Classification/embedding/save")
        self.tfidf_path = os.path.join(self.model_path, "tf_idf.bin")
        self.w2v_path = os.path.join(self.model_path, "w2v.bin")
        self.fasttext_path = os.path.join(self.model_path, "fast.bin")
        self.train_data_path = "/Users/wangruiqian/Documents/数据/图书分类数据集/train_clean.csv"
        self.dev_data_path = "/Users/wangruiqian/Documents/数据/图书分类数据集/dev_clean.csv"
        self.test_data_path = "/Users/wangruiqian/Documents/数据/图书分类数据集/test_clean.csv"
        self.stop_words_path = '/Users/wangruiqian/Documents/Code/Project/Book_Classification/data/hit_stopword'
        self.pad_size = 32
        self.batch_size = 256
        self.is_shuffle = True
        self.learn_rate = 0.001
        self.num_epochs = 100
        self.merge_file_no_stopword_csv = "/Users/wangruiqian/Documents/Code/Project/Book_Classification/data/merge_data_no_stopword.csv"
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')