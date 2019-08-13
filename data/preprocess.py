import pandas as pd
from config import opt


def preprocess():
    train_set()
    val_test_set()


def train_set():
    dataset_name = './data/train.csv'
    trainSet = load_dataset(dataset_name)

    trainSet.to_csv('./data/trainSet.csv', header=False, index=False, sep=' ')


def val_test_set():
    dataset_name = './data/valid.csv'
    dataset = load_dataset(dataset_name)
    nrow = len(dataset)
    val_row = int(nrow / 2)
    validSet = dataset[0: val_row]
    testSet = dataset[val_row: nrow]

    validSet.to_csv('./data/validSet.csv', header=False, index=False, sep=' ')
    testSet.to_csv('./data/testSet.csv', header=False, index=False, sep=' ')


def load_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name)

    class_names = opt.classes

    columes = ['Path'] + class_names

    dataset = dataset[columes].fillna(0)
    dataset = dataset.replace(-1, 1)
    return dataset


if __name__ == '__main__':
    preprocess()

