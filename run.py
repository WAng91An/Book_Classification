from src.data_process.load_data import *
from src.utils.config import Const

const = Const()

def main():
    saver(const.merge_file_no_stopword_csv)

if __name__ == '__main__':
    main()