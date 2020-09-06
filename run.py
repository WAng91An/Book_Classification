from data_proecess.embedding import *
from data_proecess.load_data import *
from common.const import Const

const = Const()

def main():
    saver(const.merge_file_no_stopword_csv)

if __name__ == '__main__':
    main()