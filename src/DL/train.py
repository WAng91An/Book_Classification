import time
import torch
import numpy as np
import pandas as pd
from importlib import import_module
import argparse
from torch.utils.data import DataLoader
import joblib
from src.data_process.DLDataset import MyDataset, collate_fn
from src.DL.train_helper import train, init_network
from src.data_process.dictionary import Dictionary
from src.utils import config
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument(
    '--model',
    type=str,
    default='CNN',
    # required=True,
    help='choose a model: CNN, RNN, RCNN, RNN_Att, DPCNN, Transformer')
parser.add_argument('--word',
                    default=True,
                    type=bool,
                    help='True for word, False for char')
parser.add_argument('--max_length',
                    default=128,
                    type=int,
                    help='True for word, False for char')
parser.add_argument('--dictionary',
                    default=None,
                    type=str,
                    help='dictionary path')
args = parser.parse_args()

if __name__ == '__main__':

    config.dl_model_name = args.model

    x = import_module('models.' + config.dl_model_name)

    # bert 模型加载 tokenizer 和 设置超参数
    if config.dl_model_name in ['bert', 'xlnet', 'roberta']:

        config.bert_path = config.root_path + '/model/' + config.dl_model_name + '/'

        if 'bert' in config.dl_model_name:
            config.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        elif 'xlnet' in config.dl_model_name:
            config.tokenizer = XLNetTokenizer.from_pretrained(config.bert_path)
        elif 'roberta' in config.dl_model_name:
            config.tokenizer = RobertaTokenizer.from_pretrained(config.bert_path)
        else:
            raise NotImplementedError

        config.save_path = config.root_path + '/model/saved_dict/' + config.dl_model_name + '.ckpt'  # 模型训练结果
        config.log_path = config.root_path + '/logs/' + config.dl_model_name
        config.hidden_size = 768
        config.eps = 1e-8
        config.gradient_accumulation_steps = 1
        config.word = True
        config.max_length = 400

    # 结果复现，保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    # 读取数据
    data = pd.read_csv(config.train_clean_path).dropna()
    if args.word:
        data = data['text'].values.tolist()
    else:
        data = data['text'].apply(lambda x: " ".join("".join(x.split())))

    # 生成字典
    if args.dictionary is None:

        dictionary = Dictionary(min_count=5)
        dictionary.build_dictionary(data)
        del data
        joblib.dump(dictionary, config.root_path + '/model/vocab.bin')

    else:

        dictionary = joblib.load(args.dictionary)

    if not args.model.isupper():
        tokenizer = config.tokenizer
    else:
        tokenizer = None

    # dataset and dataloader
    train_dataset = MyDataset(config.train_clean_path,
                              dictionary,
                              args.max_length,
                              tokenizer=tokenizer,
                              word=args.word)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    dev_dataset = MyDataset(config.dev_clean_path,
                            dictionary,
                            args.max_length,
                            tokenizer=tokenizer,
                            word=args.word)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collate_fn)
    test_dataset = MyDataset(config.test_clean_path,
                             dictionary,
                             args.max_length,
                             tokenizer=tokenizer,
                             word=args.word)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn)

    # conf.n_vocab = dictionary.max_vocab_size

    model = x.Model(config).to(config.device)

    if config.dl_model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    train(config, model, train_dataloader, dev_dataloader, test_dataloader)
