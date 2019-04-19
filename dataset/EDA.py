# -*- coding:utf-8 -*-

import pandas as pd
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')

root = '/media/gfx/data1/DATA/Kaggle/iMet'
df = pd.read_csv(os.path.join(root, 'train.csv'))
ids = df['attribute_ids']
if not os.path.exists('./figs/'):
    os.makedirs('./figs')
'''
# num_classes = 1103
labels = []
for id in ids:
    for label in id.split(' '):
        labels.append(int(label))
num_classes = len(set(labels))
print('num_classes: {}'.format(num_classes))

# num_images of per id
num_ims_per_id = [0]*num_classes
for id in ids:
    for label in id.split(' '):
        num_ims_per_id[int(label)] += 1
x1 = [i for i in range(num_classes)]
plt.plot(x1, num_ims_per_id, color='coral')
plt.xlabel('label', fontsize=14)
plt.ylabel('num_images', fontsize=14)
plt.title('num_images of per attribute', fontsize=18)
plt.savefig('./figs/num_ims_per_attribute.jpg')
'''
# '''
# num of tags distribution
nums = [0]*11
for id in ids:
    num = len(id.split(' '))
    nums[num-1] += 1
print(nums)
x2 = [i for i in range(len(nums))]
rects = plt.bar(x2, nums, color='rgby')
index=[float(c)+0.4 for c in x2]
plt.ylim(ymax=max(nums)+2000, ymin=0)
plt.xticks(x2, [i+1 for i in x2])
plt.xlabel('num_tags')
plt.ylabel('num_images')
plt.title('number of tags distribution', fontsize=18)
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
plt.savefig('./figs/num_of_tags_distribution.jpg')
# '''
