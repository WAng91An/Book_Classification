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

stop_words_path = os.path.join(root_path,'data/hit_stopword')
corpus_data_file = os.path.join(root_path, "data/corpus.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/train.csv"
dev_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/dev.csv"
test_data_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/test.csv"
book_cover_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/book_cover/"

train_clean_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/train_clean.csv"
dev_clean_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/dev_clean.csv"
test_clean_path = "/Users/wangruiqian/Documents/数据/京东图书数据集/test_clean.csv"

bert_path = "/Users/wangruiqian/Documents/Code/Project/Book_Classification/model/bert"
roberta_path = "/Users/wangruiqian/Documents/Code/Project/Book_Classification/model/roberta"
xlnet_path = "/Users/wangruiqian/Documents/Code/Project/Book_Classification/model/xlnet"

class_list = [x.strip() for x in open(root_path + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
num_classes = len(class_list)

num_epochs = 30  # epoch数
batch_size = 32  # mini-batch大小
pad_size = 400  # 每句话处理成的长度(短填长切)
learning_rate = 2e-5  # 学习率
dropout = 1.0  # 随机失活
require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
n_vocab = 50000  # 词表大小，在运行时赋值
embed = 300  # 向量维度
hidden_size = 512  # lstm隐藏层
num_layers = 1  # lstm层数
eps = 1e-8
max_length = 400
dim_model = 300
hidden = 1024
last_hidden = 512
num_head = 5
num_encoder = 2

filter_sizes = (2, 3, 4)
num_filters = 256

# explain ai
model_type = 'bert'
max_seq_length = 250
do_lower_case = True
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 1
gradient_accumulation_steps = 1
# learning_rate = 5e-5
weight_decay = 1.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 5.0
max_steps = -1
warmup_steps = 0
start_pos = 0
end_pos = 2000
visualize = -1
seed = 111

dl_model_name = "bert"

