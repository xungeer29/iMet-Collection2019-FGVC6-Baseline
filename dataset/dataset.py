# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Kaggle_iMet')
import os
import cv2
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from .transforms import tensor_transform

def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            ims.append(im)
            labels.append(int(label))
    return ims, labels


class iMetTrainDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.ims = df['id']
        self.labels = df['attribute_ids']
        self.transform = transform

    def __getitem__(self, index):
        im_name = self.ims[index]
        im_path = os.path.join(config.data_root, 'train', im_name+'.png')
        label = self.labels[index]
        target = torch.zeros(config.num_classes)
        im = Image.open(im_path)
        if self.transform is not None:
            im = self.transform(im)

        for id in label.split(' '):
            target[int(id)] = 1
        im = tensor_transform(im)

        return im, target

    def __len__(self):
        return len(self.ims)

class iMetTestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.ims = df['id']
        self.transform = transform

    def __getitem__(self, index):
        im_name = self.ims[index]
        im_path = os.path.join(config.data_root, 'test', im_name+'.png')
        im = Image.open(im_path)
        if self.transform is not None:
            im = self.transform(im)
        im = tensor_transform(im)

        return im, im_name

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = TMDataset('./data/train.txt', width=256, height=256, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    #for im, loc, cls in dataloader_train:
    for data in dataloader_train:
        print(data)
        #print loc, cls
    
