import pandas as pd
from config import opt
import os
from PIL import Image
import numpy as np


def load_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name)

    class_names = opt.classes

    columes = ['Path'] + class_names

    dataset = dataset[columes].fillna(0)
    dataset = dataset.replace(-1, 1)
    return dataset


def main():
    dataset_name = 'train.csv'
    trainSet = load_dataset(dataset_name)

    image_pth = trainSet['Path'][0]
    image_pth = os.path.join(opt.data_root, image_pth)
    image = Image.open(image_pth).convert('RGB')
    print(np.size(image))


if __name__ == '__main__':
    main()