# coding:utf8
# author: Gofinge

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class ChestXrayDataSet(Dataset):

    def __init__(self, root, image_list_file, transform=None, mode='train'):
        """
        Args:
            root: root path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        imgs_path = []
        labels = []

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                img_path = os.path.join(root, items[0])
                label = items[1:]

                imgs_path.append(img_path)
                labels.append(label)

        self.imgs_path = imgs_path
        self.labels = labels

        if transform is None:
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

            if mode == 'train':
                transform = transforms.Compose([
                    transforms.Resize([320, 320]),
                    # transforms.RandomResizedCrop(320),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif mode == 'test':
                transform = transforms.Compose([
                    transforms.Resize(320),
                    transforms.TenCrop(320),
                    transforms.Lambda
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda
                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                ])

            self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its label
        """
        img_path = self.imgs_path[index]
        img = Image.open(img_path).convert('RGB')
        label = torch.FloatTensor(list(map(float, self.labels[index])))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_path)
