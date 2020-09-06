from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from utils.util import *
from common.const import Const
from gensim import models
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

const = Const()

def trainer_tfidf(path):
    """
    corpus=["我 来到 北京 清华大学",
            "他 来到 了 网易 杭研 大厦",
            "小明 硕士 毕业 与 中国 科学院",
            "我 爱 北京 天安门"]
    :param path:
    :return:
    """
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
    # 通过 joblib.dump 保存 tfidf
    tf_idf = trainer_tfidf(path)

    joblib.dump(tf_idf, const.tfidf_path)
    print('save tfidf embedding')
    # w2v 可以通过自带的 save 函数进行保存
    w2v = trainer_w2v(path)
    joblib.dump(w2v, const.w2v_path)
    print('save word2vec embedding')
    # fast 可以通过自带的 save 函数进行保存
    fast = trainer_fasttext(path)
    joblib.dump(fast, const.fasttext_path)
    print('save fast embedding')


def load_model():
    '''
    函数说明：该函数加载训练好的模型
    '''
    # 加载模型
    # tfidf 可以通过 joblib.load 进行加载
    # w2v 和 fast 可以通过 gensim.models.KeyedVectors.load 加载

    print('load tfidf_embedding embedding')
    tfidf = joblib.load(const.tfidf_path)
    print('load w2v_embedding embedding')
    w2v = joblib.load(const.w2v_path)
    print('load fast_embedding embedding')
    fast = joblib.load(const.fasttext_path)
    return tfidf, w2v, fast


# if __name__ == '__main__':
#     a = np.load("/Users/wangruiqian/Documents/数据/RSNA/hemorrhage_224_224_3/hemorrhage_slice_label/ID_0a2b6199/19dac1a16e/19.npy")