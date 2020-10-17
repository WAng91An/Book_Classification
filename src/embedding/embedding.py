from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.util import *
from src.utils import config
from gensim import models
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
import gensim

class Embedding():
    def trainer_tfidf(self, path):
        """
        https://www.cnblogs.com/caiyishuai/p/9511567.html
        corpus=["我 来到 北京 清华大学",
                "他 来到 了 网易 杭研 大厦",
                "小明 硕士 毕业 与 中国 科学院",
                "我 爱 北京 天安门"]
        TF-IDF（Term Frequency-InversDocument Frequency）是一种常用于信息处理和数据挖掘的加权技术。
        该技术采用一种统计方法，根据字词的在文本中出现的次数和在整个语料中出现的文档频率来计算一个字词在整个语料中的重要程度。
        它的优点是能过滤掉一些常见的却无关紧要本的词语，同时保留影响整个文本的重要字词。
            TF（Term Frequency）表示某个关键词在整篇文章中出现的频率。
            IDF（InversDocument Frequency）表示计算倒文本频率。文本频率是指某个关键词在整个语料所有文章中出现的次数。
            倒文档频率又称为逆文档频率，它是文档频率的倒数，主要用于降低所有文档中一些常见却对文档影响不大的词语的作用。
        （1）计算词频
    　　      词频 = 某个词在文章中出现的总次数/文章的总词数
        （2）计算逆文档频率
            逆文档频率（IDF） = log（词料库的文档总数/包含该词的文档数+1）
        """
        # corpus = get_corpus(path, tf_idf=True)
        # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        #
        # return tfidf

        corpus = get_corpus(path, tf_idf=True)
        count_vect = TfidfVectorizer(stop_words=get_stop_word_list(),
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        tfidf = count_vect.fit(corpus)
        return tfidf

    def trainer_w2v(self, path):
        """
        corpus=['教授', '长江', '学者', '优秀成果', '集中', '呈现'], ['', '生物质', '发电', '燃料', '供应链', '运营', '模式']
        :param path:
        :return:
        """
        print('train w2v')
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


    def trainer_fasttext(self, path):
        print('train fasttext')
        corpus = get_corpus(path, w2v=True)
        fast = models.FastText(corpus, size=300, window=3, min_count=2)
        return fast

    def train_lda(self, path):
        """
        https://www.cnblogs.com/Luv-GEM/p/10881838.html
        gensim 所需要的输入格式为：['教授', '长江', '学者', '优秀成果', '集中', '呈现']，也就是每段文章 text 是一个列表，元素为词语。
        然后构建语料库，再利用语料库把每篇新闻进行数字化，corpus 就是数字化后的结果。
        第一段文章 text ID 化后的结果为 corpus[0]：[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), ...]，每个元素是 text 中的每个词语的 ID 和频率。
        最后训练 LDA 模型。LDA是一种无监督学习方法，我们可以自由选择主题的个数。num_topics = 30
        """

        print('train lda')
        corpus_data = get_corpus(path, w2v=True)
        id2word = gensim.corpora.Dictionary(corpus_data)
        corpus = [id2word.doc2bow(text) for text in corpus_data]

        # corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
        # corpus是把每条ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率

        LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha='asymmetric')
        return LDAmodel

    def saver(self, path):
        '''
        函数说明：该函数存储训练好的模型
        '''
        # 通过 joblib.dump 保存 tfidf
        tf_idf = self.trainer_tfidf(path)
        joblib.dump(tf_idf, config.tfidf_path)
        print('save tfidf model')
        # w2v 可以通过自带的 save 函数进行保存
        w2v = self.trainer_w2v(path)
        joblib.dump(w2v, config.w2v_path)
        print('save word2vec model')
        # fast 可以通过自带的 save 函数进行保存
        fast = self.trainer_fasttext(path)
        joblib.dump(fast, config.fasttext_path)
        print('save fast model')

        Lda_model = self.train_lda(path)
        joblib.dump(Lda_model, config.lda_path)
        print('save lda model')

    def load_model(self):
        '''
        函数说明：该函数加载训练好的模型
        '''
        # 加载模型
        # tfidf 可以通过 joblib.load 进行加载
        # w2v 和 fast 可以通过 gensim.models.KeyedVectors.load 加载

        print('load tfidf_embedding model')
        tfidf = joblib.load(config.tfidf_path)
        print('load w2v_embedding model')
        w2v = joblib.load(config.w2v_path)
        print('load fast_embedding model')
        fast = joblib.load(config.fasttext_path)
        print('load lda model')
        lda = joblib.load(config.lda_path)
        return tfidf, w2v, fast, lda

if __name__ == "__main__":
    em = Embedding()
    # corpus_data_file 为生成的语料库的地址，该文件通过 data_process/build_corpus.py 生成。由三个训练数据生成的
    em.saver(config.corpus_data_file)