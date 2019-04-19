# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Kaggle_iMet')

import os
import random
import pandas as pd
from config import config
from tqdm import tqdm

random.seed(config.seed)

if not os.path.exists('./data'):
    os.makedirs('./data')

train_txt = open('./data/train.txt', 'w')
train_txt.write('id,attribute_ids\n')

val_txt = open('./data/valid.txt', 'w')
val_txt.write('id,attribute_ids\n')

df = pd.read_csv(os.path.join(config.data_root, 'train.csv'))

# 标签数量分布{label_num: im_num} = {1:4324, 2:37356, 3:29200, 4:20208, 5:10946,
#                                    6:6157, 7:920, 8:103, 9:17, 10:5, 11:1}
num2ims = {}
name2id = {}
for name, id in zip(df['id'], df['attribute_ids']):
    name2id[name] = id
    for i in range(1, 12):
        if len(id.split(' ')) == i:
            if str(i) not in num2ims.keys():
                num2ims[str(i)] = []
            num2ims[str(i)].append(name)

val = []
for name in random.sample(num2ims['1'], 35):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['2'], 2554):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['3'], 1550):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['4'], 747):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['5'], 219):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['6'], 67):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['7'], 5):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['8'], 3):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['9'], 2):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)
for name in random.sample(num2ims['10'], 1):
    val_txt.write('{},{}\n'.format(name, name2id[name]))
    val.append(name)

for name in df['id']:
    if name in val:
        continue
    train_txt.write('{},{}\n'.format(name, name2id[name]))
