from collections import Counter

class Dictionary(object):

    def __init__(self, max_vocab_size=50000, min_count=None, start_end_tokens=False,
                 wordvec_mode=None, embedding_size=300):

        # 定义所需要参数
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count # 低频词过滤
        self.start_end_tokens = start_end_tokens # 是否添加开始结束的 token
        self.embedding_size = embedding_size
        self.wordvec_mode = wordvec_mode
        self.PAD_TOKEN = '<PAD>'

    def build_dictionary(self, data):
        # 构建词典主方法， 使用_build_dictionary构建
        self.voacab_words, self.word2idx, self.idx2word, self.idx2count = self._build_dictionary(data)
        self.vocabulary_size = len(self.voacab_words)

        if self.wordvec_mode is None:
            self.embedding = None
        elif self.wordvec_mode == 'word2vec':
            self.embedding = self._load_word2vec()

    def indexer(self, word):
        # 根据词获取到对应的id
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_dictionary(self, data):
        # 加入UNK标示， 按照需要加入EOS 或者EOS
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2
        if self.start_end_tokens:
            vocab_words += ['<SOS>', '<EOS>']
            vocab_size += 2

        # 使用 Counter 来统计的个数
        counter = Counter([word for sentence in data for word in sentence.split()])

        # 按照最大的词典个数进行筛选
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size - vocab_size)}

        # 过滤掉低频词
        if self.min_count:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_count}

        # 按照出现次数进行排序， 并加到 vocab_words 当中
        counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        # {'本书': 132532, '中': 93875}
        vocab_words += list(counter.keys())
        # ['<PAD>', '<UNK>', '本书', '中', '中国' ... ]
        idx2count = [counter.get(word, 0) for word in vocab_words] # 查找 counter 中 key 为 word 的 values 值，若找不到则返回 0，0 对应着UNK
        # [0, 0, 132532, 93875, 78370, 70092 ... ]
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        # {'<PAD>': 0, '<UNK>': 1, '本书': 2, '中': 3 ... ]
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word, idx2count
