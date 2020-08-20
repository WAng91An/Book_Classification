from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from utils.util import *
from gensim import models
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer


def trainer_tfidf(path):

    corpus = get_corpus(path, tf_idf=True)

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵

    return tfidf
    # return tfidf.toarray()


def trainer_w2v(path):
    corpus = get_corpus(path, w2v=True)
    w2v = models.Word2Vec(min_count=2,
                          window=3,
                          size=300,
                          sample=6e-5,
                          alpha=0.03,
                          min_alpha=0.0007,
                          negative=15,
                          workers=4,
                          iter=10,
                          max_vocab_size=50000)
    w2v.build_vocab(corpus)
    w2v.train(corpus,
              total_examples=w2v.corpus_count,
              epochs=15,
              report_delay=1)

    return w2v


def trainer_fasttext(path):
    corpus = get_corpus(path, w2v=True)
    fast = models.FastText(corpus, size=300, window=3, min_count=2)
    return fast


def saver(path):
    '''
    函数说明：该函数存储训练好的模型
    '''
    # hint: 通过joblib.dump保存tfidf
    tf_idf = trainer_tfidf(path)

    joblib.dump(tf_idf, tfidf_path)
    print('save tfidf model')
    # hint: w2v可以通过自带的save函数进行保存
    w2v = trainer_w2v(path)
    joblib.dump(w2v, w2v_path)
    print('save word2vec model')
    # hint: fast可以通过自带的save函数进行保存
    fast = trainer_fasttext(path)
    joblib.dump(fast, fasttext_path)
    print('save fast model')


def load_model():
    '''
    函数说明：该函数加载训练好的模型
    '''
    # ToDo
    # 加载模型
    # hint: tfidf可以通过joblib.load进行加载
    # w2v和fast可以通过gensim.models.KeyedVectors.load加载
    print('load tfidf_embedding model')
    tfidf = joblib.load(tfidf_path)
    print('load w2v_embedding model')
    w2v = joblib.load(w2v_path)
    print('load fast_embedding model')
    fast = joblib.load(fasttext_path)
    return tfidf, w2v, fast


# if __name__ == '__main__':
#     a = np.load("/Users/wangruiqian/Documents/数据/RSNA/hemorrhage_224_224_3/hemorrhage_slice_label/ID_0a2b6199/19dac1a16e/19.npy")