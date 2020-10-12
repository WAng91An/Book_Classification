# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from src.utils.util import *
from src.utils.config import Config
from src.embedding.embedding import load_model
from sklearn import preprocessing
from sklearn.decomposition import PCA

# from bayes_opt import BayesianOptimization

const = Const()

max_length = 500  # 表示样本表示最大的长度,表示降维之后的维度
sentence_max_length = 1500  # 表示句子/样本在降维之前的维度

# Train_features3, Test_features3, Train_label3, Test_label3 = [], [], [], []
tfidf_model, w2v_embedding, fast_embedding, train, test = None, None, None, None, None

def load():

    global tfidf_model, w2v_embedding, fast_embedding
    tfidf_model, w2v_embedding, fast_embedding = load_model()

    #print(fast_embedding["领域"]) # 查看 领域 这个词在 fasttext 下的 model 结果
    #print(w2v_embedding["领域"])  # 查看 领域 这个词在 word2vec 下的 model 结果

    print("fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}".format(len(fast_embedding.wv.vocab.keys()), len(w2v_embedding.wv.vocab.keys())))

    print("取词向量成功")

def gen_label_index_mapping():
    """
    为原始的数据集增加了一列，这一列是 label 名字对应的索引
    :return:
    """
    global train, test

    train = pd.read_csv(const.train_data_path, sep="\t")
    test = pd.read_csv(const.test_data_path, sep="\t")

    print("读取数据完成")

    label_name = train.label.unique()

    label_index = list(range(len(label_name)))  # 全部label标签
    label_name_to_index = dict(zip(label_name, label_index))  # label的名字对应标签的字典
    # {'文化': 0, '文学': 1, '管理': 2, '社会科学': 3 ...}
    label_index_to_name = dict(zip(label_index, label_name))  # label的标签对应名字的字典
    # {0: '文化', 1: '文学', 2: '管理', 3: '社会科学' ...}

    train["labelIndex"] = train.label.map(label_name_to_index)
    test["labelIndex"] = test.label.map(label_name_to_index)

def query_cut(query):
    '''
    函数说明：该函数用于对输入的语句（query）按照空格进行切分
    '''
    query_list = query.split(" ")
    return query_list

def gen_query_cut():
    global train, test
    """
    然后train和test中的每一个样本都按照空格进行切分
    并将划分好的样本分别存储到train["queryCut"]和test["queryCut"]中
    :return:
    """
    print("切分数据中...")
    train_text = []
    for i in range((len(train.text))):
        train_text.append(query_cut(train.text[i]))

    test_text = []
    for i in range((len(test.text))):
        test_text.append(query_cut(test.text[i]))

    train["queryCut"] = train_text
    test["queryCut"] = test_text

    print("切分数据完成")

def rm_stop_word():
    global train, test
    '''
    函数说明：该函数用于去除输入样本中的存在的停用词
    Return: 返回去除停用词之后的样本
    '''
    # 第一步：按行读取停用词文件
    # 第二步：去除每个样本中的停用词并返回新的样本
    print("去除停用词中...")
    def remove_stop_word(wrod_list):
        stop_word_list = get_stop_word_list()
        line_text = ""
        for word in wrod_list:
            if word not in stop_word_list:
                line_text = line_text + " " + word
        return line_text

    train["queryCutRMStopWord"] = train["queryCut"].apply(remove_stop_word)
    test["queryCutRMStopWord"] = test["queryCut"].apply(remove_stop_word)
    print("去除停用词完成")


# def Find_embedding_with_windows(embedding_matrix, mean=True, k=(2, 3, 4)):
#     '''
#     函数说明：该函数用于获取在大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
#     参数说明：
#        - embedding_matrix：样本中所有词构成的词向量矩阵
#     return: result_list 返回拼接而成的一维词向量
#     '''
#     # embedding_matrix (153, 300)
#     length = len(embedding_matrix)
#     if length == 1:
#         result_list = embedding_matrix
#     else:
#         result_list = []
#         for _k in k:
#             for i in range(length - _k + 1):
#                 cur_embedding = np.mean(embedding_matrix[i:i + _k], axis=0)
#                 # print("cur_embedding", cur_embedding.shape)
#                 result_list.append(cur_embedding)
#
#     # 由于之前抽取的特征并没有考虑词与词之间交互对模型的影响，
#     # 对于分类模型来说，贡献最大的不一定是整个句子， 可能是句子中的一部分， 如短语、词组等等。
#     # 在此基础上我们使用大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
#     if mean:
#         result_list = np.mean(result_list, axis=0)
#     else:
#         result_list = np.max(result_list, axis=0)
#     return result_list

def Find_embedding_with_windows(embedding_matrix, window_size=2, method='mean'):
    '''
    @description: generate model use window
    @param {type}
    embedding_matrix, input sentence's model # (153, 300)
    window_size, 2, 3, 4
    method, max/ mean
    @return: ndarray of model
    '''
    # 最终的词向量
    result_list = []
    # 遍历input的长度， 根据窗口的大小获取embedding， 进行mean操作， 然后将得到的结果extend到list中， 最后进行mean max 聚合
    for k1 in range(len(embedding_matrix)):
        # 如何当前位置 + 窗口大小 超过input的长度， 则取当前位置到结尾
        # mean 操作后要reshape 为 （1， 300）大小
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.extend(np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
        else:
            result_list.extend(np.mean(embedding_matrix[k1:k1 + window_size], axis=0).reshape(1, 300))
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def max_pooling(embedding_matrix, window_size):
    # creates a matrix where each row is the result of 1 step of window_size
    shape = embedding_matrix.shape[:-1] + (embedding_matrix.shape[-1] - window_size + 1, window_size)
    strides = embedding_matrix.strides + (embedding_matrix.strides[-1],)
    matrix = np.lib.stride_tricks.as_strided(embedding_matrix, shape=shape, strides=strides)

    result = []
    for i in range(len(matrix)):
        result.append(np.max(matrix[i]))
    return result

def softmax(x):
    '''
    @description: calculate softmax
    @param {type}
    x, ndarray of model
    @return: softmax result
    '''
    return np.exp(x) / np.exp(x).sum(axis=0)

def Find_Label_embedding(example_matrix, label_embedding, method="mean"):
    '''
    函数说明：获取到所有类别的 label model， 与输入的 word model 矩阵相乘， 对其结果进行 softmax 运算，
            对 attention score 与输入的 word model 相乘的结果求平均或者取最大
            可以参考论文《Joint model of words and labels》获取标签空间的词嵌入
    parameters:
    -- example_matrix(np.array 2D): denotes the matrix of words model
    -- model(np.array 2D): denotes the model of all label in data
    return: (np.array 1D) the model by join label and word
    '''

    # 根据矩阵乘法来计算label与word之间的相似度
    #print("example_matrix", example_matrix.shape) # (153, 300)
    #print("label_embedding", label_embedding.shape) # (2, 300)
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))

    #print("similarity_matrix", similarity_matrix.shape) # (153, 2)
    # np.linalg.norm: 矩阵整体元素平方和开根号，不保留矩阵二维特性
    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    # 这里可以使用max-pooling和mean-pooling

    attention = similarity_matrix.max(axis=1)
    #print(attention.shape) # (153, )
    attention = softmax(attention)
    #print(attention.shape) # (153, )
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention.reshape(attention.shape[0], 1)
    #print(attention_embedding.shape) #  (153, 300)
    if method == 'mean':
        #print(np.mean(attention_embedding, axis=0).shape) #  (300)
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)


def sentence2vec(query, label):
    '''
    函数说明：联合多种特征工程来构造新的样本表示，主要通过以下三种特征工程方法
            第一：利用word-embedding的average pooling和max-pooling
            第二：利用窗口size=2，3，4对word-embedding进行卷积操作，然后再进行max/avg-pooling操作
            第二：利用类别标签的表示，增加了词语和标签之间的语义交互，以此达到对词级别语义信息更深层次的考虑
            另外，对于词向量超过预定义的长度则进行截断，小于则进行填充
    参数说明：query:数据集中的每一个样本
    return: 返回样本经过特征工程之后得到的词向量
    '''
    # 加载 fast_embedding, w2v_embedding
    global fast_embedding, w2v_embedding
    # 将一句话中的每个词转化成 vector，然后合并成 array
    fast_arr = np.array([fast_embedding.wv.get_vector(s) for s in query if s in fast_embedding.wv.vocab.keys()])
    label_embedding = np.array([fast_embedding.wv.get_vector(s) for s in label if s in fast_embedding.wv.vocab.keys()])

    #print("query", query) # query  杏林 芳菲 广东 中医药 曹磊 编著 杏林 芳菲 广东 中医药 注重 文化 内涵 挖掘 特殊 技艺 介绍 非 物质 文化遗产 广东 中医药 内涵 技艺 形态 历史 演变 艺术 价值 给予 全面 介绍 深刻 直观 记录 时代 变迁 记录 民间 丰富 生活 图文并茂 生动活泼 富有 艺术 表现力 读者 文化 审美 感受
    #print("label", label) # label 文化
    #print(fast_arr.shape) # (153, 300)
    #print(label_embedding.shape) # (2, 300)

    # 在 fast_arr 下滑动获取到的词向量
    if len(fast_arr) > 0:
        windows_fastarr = np.array(Find_embedding_with_windows(fast_arr))
        result_attention_embedding = Find_Label_embedding(fast_arr, label_embedding)
    else:
        # 如果样本中的词都不在字典，则该词向量初始化为0
        # 这里300表示训练词嵌入设置的维度为300
        windows_fastarr = np.zeros(300)
        result_attention_embedding = np.zeros(300)

    fast_arr_max = np.max(np.array(fast_arr), axis=0) if len(fast_arr) > 0 else np.zeros(300)
    fast_arr_avg = np.mean(np.array(fast_arr), axis=0) if len(fast_arr) > 0 else np.zeros(300)

    #print("fast_arr_max", fast_arr_max.shape)
    #print("fast_arr_avg", fast_arr_avg.shape)

    fast_arr = np.hstack((fast_arr_avg, fast_arr_max))
    # 将多个embedding进行横向拼接
    arr = np.hstack((np.hstack((fast_arr, windows_fastarr)), result_attention_embedding))
    global sentence_max_length
    # 如果样本的维度大于指定的长度则需要进行截取或者拼凑,
    result_arr = arr[:sentence_max_length] if len(arr) > sentence_max_length else np.hstack((arr, np.zeros(int(sentence_max_length-len(arr)))))
    return result_arr

def Dimension_Reduction(Train, Test):
    '''
    函数说明：该函数通过PCA算法对样本进行降维，由于目前维度不是特别搞高 ，可以选择不降维。
    参数说明：
    - Train: 表示训练数据集
    - Test: 表示测试数据集
    Return: 返回降维之后的数据样本
    '''
    global max_length
    # To_Do
    # 特征选择，由于经过特征工程得到的样本表示维度很高，因此需要进行降维 max_length 表示降维之后的样本最大的维度。
    # 这里通过PCA方法降维
    pca = PCA(n_components=max_length, svd_solver='full')
    pca_train = pca.fit_transform(Train)
    pca_test = pca.fit_transform(Test)
    return pca_train, pca_test


def Find_Embedding():
    '''
    函数说明：该函数用于获取经过特征工程之后的样本表示
    将每个句子进行特征工程得到一个 sentence vector，然后再将这些 vector 合并起来后进行归一化
    # 将归一化后的结果进行降维
    Return:训练集特征数组(2D)，测试集特征数组(2D)，训练集标签数组（1D）,测试集标签数组（1D）
    '''
    global train, test
    print(train)
    print("获取样本表示中...")
    min_max_scaler = preprocessing.MinMaxScaler() # MinMaxScaler：归一到 [ 0，1 ]
    print("sentence2vec...")
    Train_features = min_max_scaler.fit_transform(np.vstack(train.apply(lambda row: sentence2vec(row['queryCutRMStopWord'], row['label']), axis=1)))
    Test_features = min_max_scaler.fit_transform(np.vstack(test.apply(lambda row: sentence2vec(row['queryCutRMStopWord'], row['label']), axis=1)))
    print("获取样本词表示完成")
    print("降维前 train 的形状：", Train_features.shape)
    print("降维前 train 的形状：", Test_features.shape)
    print(Test_features.shape)
    # 通过PCA对样本表示进行降维
    Train_features, Test_features = Dimension_Reduction(Train=Train_features, Test=Test_features)
    Train_label = train["labelIndex"]
    Test_label = test["labelIndex"]

    print("Train_features.shape =", Train_features.shape)
    print("Test_features.shape =", Test_features.shape)
    print("Train_label.shape =", Train_label.shape)
    print("Test_label.shape =", Test_label.shape)

    return Train_features, Test_features, Train_label, Test_label


def Predict(Train_label, Test_label, Train_predict_label, Test_predict_label, model_name):
    '''
    函数说明：直接输出训练集和测试在模型上的准确率
    参数说明：
        - Train_label: 真实的训练集标签（1D）
        - Test_labelb: 真实的测试集标签（1D）
        - Train_predict_label: 模型在训练集上的预测的标签(1D)
        - Test_predict_label: 模型在测试集上的预测标签（1D）
        - model_name: 表示训练好的模型
    Return: None
    '''
    # ToDo
    # 通过调用metrics.accuracy_score计算训练集和测试集上的准确率
    # print(Search_Flag+model_name+'_'+'Train accuracy %s' % #Todo )
    # # 输出测试集的准确率
    # print(Search_Flag+model_name+'_'+'test accuracy %s' % #Todo )

def Grid_Train_model(Train_features, Test_features, Train_label, Test_label):
    '''
    函数说明：基于网格搜索优化的方法搜索模型最优参数，最后保存训练好的模型
    参数说明：
        - Train_features: 训练集特征数组（2D）
        - Test_features: 测试集特征数组（2D）
        - Train_label: 真实的训练集标签 (1D)
        - Test_label: 真实的测试集标签（1D）
    Return: None
    '''
    lgb_parameters = {
        'max_depth': [25],
        'learning_rate': [0.1, 0.15],
        'n_estimators': [100, 500],
        'min_child_weight': [5, 10],
        'max_delta_step': [1, 2],
        'subsample': [0.6, 0.7, ],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.75, 1],
        'reg_lambda': [0.4, 0.6],
        'scale_pos_weight': [0.2, 0.4]
    }
    # 定义分类模型列表，这里仅使用LightGBM模型

    svc_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
    nb_parameters = {'class_prior': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]}
    lr_parameters = {'C': [10 ** i for i in range(-4, 4, 2)],  # 指数分布
                     'multi_class': ['ovr', 'multinomial']}
    models = {
        lgb.LGBMClassifier(): lgb_parameters,
        SVC(): svc_parameters,
        MultinomialNB(): nb_parameters,
        LogisticRegression(penalty='l2', solver='lbfgs', tol=1e-6): lr_parameters
    }

    # 遍历模型
    for model, parameters in models.items():
        model_name = model.__class__.__name__
        gsearch = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
        gsearch.fit(Train_features, Train_label)

        # 输出最好的参数
        print("Best parameters set found on development set:{}".format(gsearch.best_params_))

        Test_predict_label = gsearch.predict(Test_features)
        Train_predict_label = gsearch.predict(Train_features)

        Predict(Train_label, Test_label,Train_predict_label, Test_predict_label, model_name)
    # 保存训练好的模型
    # joblib.dump('#ToDo' + '.pkl')

if __name__ == '__main__':

    load()
    gen_label_index_mapping()
    gen_query_cut()
    rm_stop_word()

    Train_features, Test_features, Train_label, Test_label = Find_Embedding()
    Grid_Train_model(Train_features=Train_features, Test_features=Test_features, Train_label=Train_label, Test_label=Test_label)