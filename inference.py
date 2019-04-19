# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Kaggle_iMet')
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from dataset.transforms import test_transform
from utils.plot import *
from config import config
from metrics.metric import * 

def read_label(path):
    data = open(path, 'r')
    label2name = {}
    for line in data.readlines():
        name, label = line.strip().split(' ')
        label2name[int(label)] = name
    return label2name

def inference():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    # model
    # load checkpoint
    model = torch.load(os.path.join('./checkpoints', config.checkpoint))
    model.cuda()

    # test data
    dst_test = iMetTestDataset(os.path.join(config.data_root, 'sample_submission.csv'), 
                                transform=test_transform)
    dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=config.batch_size/2, 
                                  num_workers=config.num_workers)
    all_predictions = []
    all_names = []
    model.eval()
    with torch.no_grad():
        for ims, name in tqdm(dataloader_test):
            input = Variable(ims).cuda()
            output = model(input)
        
            predictions = torch.sigmoid(output)
            all_predictions.append(predictions.cpu().numpy())
            all_names += name

    all_predictions = np.concatenate(all_predictions)
    argsorted = all_predictions.argsort(axis=1)
    preds = binarize_prediction(all_predictions, args.threshold, argsorted)

    sub = open('submission.csv', 'w')
    writer = csv.writer(sub)
    writer.writerow(['id', 'attribute_ids'])
    names = []
    results = []
    for name, pred in zip(all_names, preds):
        label = []
        for idx, p in enumerate(pred):
            if p == 1:
                label.append(idx)
        writer.writerow([name, ' '.join(str(i) for i in label)])
    sub.close()
if __name__ == '__main__':
    inference()
