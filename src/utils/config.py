# -*- coding: utf-8 -*-
import os
import re
import torch

root_path = os.path.split(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])[0]
model_path = os.path.join(root_path, "model/save/")
tfidf_path = os.path.join(model_path, "tf_idf.bin")
w2v_path = os.path.join(model_path, "w2v.bin")
fasttext_path = os.path.join(model_path, "fast.bin")
lda_path = os.path.join(model_path, "lda.bin")

train_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/train.csv"
dev_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/dev.csv"
test_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/test.csv"
book_cover_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/book_cover/"

stop_words_path = os.path.join(root_path,'data/hit_stopword')
pad_size = 32
batch_size = 256
is_shuffle = True
learn_rate = 0.001
num_epochs = 100
corpus_data_file = os.path.join(root_path, "data/corpus.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')