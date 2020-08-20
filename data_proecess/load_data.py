import numpy as np
import pandas as pd
import sys
# from gensim import models
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.externals import joblib

def load_data():
    '''
    函数说明：该函数用于加载数据集
    return:
        -data: 表示所有数据拼接的原始数据
        -data_text: 表示数据集中的特征数据集
        -datatext: 表示经过分词之后的特征数据集
        -stopWords: 表示读取的停用词
    '''
    print('load Pre_process')
    data = pd.concat([
        # 该读文件方式，默认是以逗号作为分割符，若是以其它分隔符，比如制表符“/t”，则需要显示的指定分隔符。
        pd.read_csv('/Users/wangruiqian/Documents/数据/图书分类数据集/train_clean.csv', sep='\t'),
        pd.read_csv('/Users/wangruiqian/Documents/数据/图书分类数据集/dev_clean.csv', sep='\t'),
        pd.read_csv('/Users/wangruiqian/Documents/数据/图书分类数据集/test_clean.csv', sep='\t')
        ])
    print("读取数据集完成")
    data_text = list(data.text)  # .apply(lambda x: x.split(' '))
    datatext = []
    for i in range(len(data_text)):
        datatext.append(data_text[i].split(' '))
    stopWords = open('/Users/wangruiqian/Documents/Code/Project/Book_Classification/data/hit_stopword').readlines()
    print("取停用词完成")
    return data, data_text,datatext, stopWords

if __name__ == '__main__':
    load_data()