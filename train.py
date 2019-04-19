# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Kaggle_iMet')
import os, argparse, time

import numpy as np
import cv2
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from networks.lr_schedule import *
from metrics.metric import *
# from utils.plot import *
from config import config
from utils.label_smooth import *
from dataset.transforms import train_transform, test_transform


def train():
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print(model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # freeze layers
    if config.freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        #for p in model.backbone.layer4.parameters(): p.requires_grad = False


    # loss
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

    # label smoothing
    label_smoothing = LabelSmoothing(smoothing=0.05)

    # train data
    dst_train = iMetTrainDataset(os.path.join(config.data_root, 'train.csv'), 
                                 transform=train_transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, 
                                  num_workers=config.num_workers)

    # valid data
    dst_val = iMetTrainDataset(os.path.join('./data', 'valid.csv'),
                                 transform=train_transform)
    dataloader_val = DataLoader(dst_val, shuffle=False, batch_size=config.batch_size, 
                                  num_workers=config.num_workers)
    
    # test data
    dst_test = iMetTestDataset(os.path.join(config.data_root, 'sample_submission.csv'), 
                                transform=test_transform)
    dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=int(config.batch_size/2), 
                                  num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\nOHEM:{}\nOHEM_ratio:{}\nlabel_smoothing:{}'.format(
               config.model, config.num_classes, config.num_epochs, config.lr, 
               config.width, config.height, config.iter_smooth, config.OHEM, 
               config.OHEM_ratio, config.smooth_label))

    # load checkpoint
    if config.resume:
        print('resume checkpoint...')
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    # train
    sum = 0
    train_loss_sum = 0
    lr = 0.001 # config.lr
    best_f2 = 0
    for epoch in range(config.init_epoch, config.num_epochs):
        metrics = eval(model, dataloader_val, criterion)
        print(' | '.join('{} {:.3f}'.format(k, v) for k, v in sorted(
              metrics.items(), key=lambda kv: -kv[1])))
        previous_f2 = metrics['valid_f2_th_0.1']
        ep_start = time.time()

        # adjust lr
        # lr = half_lr(config.lr, epoch)
        # lr = step_lr(epoch)
        #lr = config.lr / (5 * (epoch+1))
        # lr = 0.0001

        # optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda()

            output = model(input)

            if config.smooth_label:
                smoothed_target = label_smoothing(output, target).cuda()
                loss = F.kl_div(output, smoothed_target).cuda()
            
            # OHEM: online hard example mining
            if not config.OHEM and not config.smooth_label:
                loss = criterion(output, target).sum()
            elif config.OHEM:
                if epoch < 50:
                    loss = criterion(output, target)
                else:
                    loss = F.cross_entropy(output, target, reduce=False).cuda()
                    OHEM, _ = loss.topk(int(config.num_classes*config.OHEM_ratio), dim=0, 
                                        largest=True, sorted=True)
                    loss = OHEM.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.data.cpu().numpy()
            sum += 1

            if (i+1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, best_f2: %.4f'
                       %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                       lr, train_loss_sum/sum, best_f2))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f\n'
                           %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                           lr, train_loss_sum/sum))
                sum = 0
                train_loss_sum = 0

        
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < config.num_epochs+1:
            # eval
            val_time_start = time.time()
            metrics = eval(model, dataloader_val, criterion)
            if metrics['valid_f2_th_0.1'] < previous_f2 + 0.03:
                lr /= 5
            if metrics['valid_f2_th_0.1'] > best_f2:
                best_f2 = metrics['valid_f2_th_0.1']
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}_python3.pth'.format('checkpoints', config.model))

            val_time = (time.time() - val_time_start) / 60.

            print('epoch time: {} min'.format(epoch_time))
            print(' | '.join('{} {:.3f}'.format(k, v) for k, v in sorted(
                    metrics.items(), key=lambda kv: -kv[1])))
            log.write(' | '.join('{} {:.3f}'.format(k ,v) for k, v in sorted(
                       metrics.items(), key=lambda kv: -kv[1])))
        
    log.write('-'*30+'\n')
    log.close()

# validation
def eval(model, dataloader_val, criterion):
    sum = 0
    val_loss_sum = 0
    all_predictions, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for ims, label in dataloader_val:
            all_targets.append(label.numpy().copy())

            input_val = Variable(ims).cuda()
            target_val = Variable(label).cuda()
            output_val = model(input_val)
            loss = criterion(output_val, target_val).sum()
            sum += 1
            val_loss_sum += loss.data.cpu().numpy()
        
            predictions = torch.sigmoid(output_val)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    avg_loss = val_loss_sum / sum

    # compute F2 score    
    def get_score(y_pred):
        return fbeta_score(all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        metrics['valid_f2_th_'+str(threshold)] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = avg_loss


    return metrics

if __name__ == '__main__':
    train()
