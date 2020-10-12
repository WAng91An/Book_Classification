import pandas as pd
from src.utils.util import *
from src.utils.config import *
# from gensim import models
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.externals import joblib

const = Const()

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
        # 一行分割不同的列是按照制表符分割的
        # 如果此处不指定 sep='\t' ，会报错 pandas.errors.ParserError: Error tokenizing data. C error: Expected 7 fields in line 4, saw 8
        pd.read_csv(const.train_data_path, sep='\t'),
        pd.read_csv(const.dev_data_path, sep='\t'),
        pd.read_csv(const.test_data_path, sep='\t')
        ])
    print("读取数据集完成")
    print("data", data)
    data_text = list(data.text)  # .apply(lambda x: x.split(' '))
    print("data_text", data_text)
    datatext = []
    for i in range(len(data_text)):
        datatext.append(data_text[i].split(' '))
    print("datatext", datatext)
    data_stop_list = get_stop_word_list()
    print("取停用词完成")
    return data, data_text,datatext, data_stop_list

def write_csv(datatext, stopWords):
    """
    将三个数据文件的 text 合并起来到一个文件，并且去除停用词
    :param datatext: 三个数据文件合并后的list，没有去除停用词
    :param stopWords: 包含停用词的list
    :return: merge_file_no_stopword_csv, 三个数据文件的 text 合并起来到一个文件，每一行都是一个句子（去掉了体用词，用空格分隔）
    """
    print(stopWords)

    with open(const.merge_file_no_stopword_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([["description: merge three csv file text, delete stop words"]])
        for item in datatext:
            line_text = ""
            for word in item:
                if word not in stopWords:
                    line_text = line_text + " " + word
            print(line_text)
            writer.writerows([[line_text]])

        csvfile.close()

if __name__ == '__main__':
    # 将提供的三个数据文件，train，test，dev 合并起来。并获取到停用词
    data, data_text, datatext, stopWords = load_data()
    # 将三个数据文件中 text 部分合并到一个 csv，并且去除掉停用词
    write_csv(datatext, stopWords)
