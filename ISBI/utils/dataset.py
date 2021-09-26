import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import glob


# 自定义数据集的loader
class CustomDataset(Dataset):
    def __init__(self, data_path):
        # 1. Initialize file paths or a list of file names.
        self.data_path = data_path
        self.images_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        pass

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        # 获取图片的路径
        image_path = self.images_path[index]
        label_path = image_path.replace('image', 'label')
        # label_path = image_path.replace('image', 'label_in_50')
        # label_path = image_path.replace('image', 'label_in_50')
        # label_path = image_path.replace('image', 'label_incomplete')

        # 获取图片 im read 接口读取进来为 0~255 范围的 BGR格式的数据
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 图片处理过程 转化为单通道 （C=1 H W）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image_h = image.shape[0]
        image_w = image.shape[1]
        label_h = label.shape[0]
        label_w = label.shape[1]
        image = image.reshape(1, image_h, image_w)
        label = label.reshape(1, label_h, label_w)

        # 标签像素归一化处理？
        if label.max() > 1:
            label = label / 255

        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.images_path)

    # You can then use the prebuilt data loader.


class ValidDataset(Dataset):
    def __init__(self, data_path):
        # 1. Initialize file paths or a list of file names.
        self.data_path = data_path
        self.images_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        pass

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        # 获取图片的路径
        image_path = self.images_path[index]
        label_path = image_path.replace('image', 'label')

        # 获取图片 im read 接口读取进来为 0~255 范围的 BGR格式的数据
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 图片处理过程 转化为单通道 （C=1 H W）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image_h = image.shape[0]
        image_w = image.shape[1]
        label_h = label.shape[0]
        label_w = label.shape[1]
        image = image.reshape(1, image_h, image_w)
        label = label.reshape(1, label_h, label_w)

        # 标签像素归一化处理？
        if label.max() > 1:
            label = label / 255

        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.images_path)

    # You can then use the prebuilt data loader.


if __name__ == '__main__':
    custom_dataset = CustomDataset("../data/train")
    print("数据个数：", len(custom_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for train_image, train_label in train_loader:
        print(train_image.shape)
        print(train_label.shape)
