#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
conda install python-graphviz
conda install -c conda-forge visdom 
'''
from __future__ import print_function
import os
import random
import time
import numpy as np
import torch
import torchvision
# from torch.utils import data
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from utils.visualizer import Visualizer
from utils import view_model
from dataset import FaceDataset 
from config import Config
from models.resnet import resnet_face18, resnet34, resnet50
from models.metrics import ArcMarginProduct
from models.focal_loss import FocalLoss
from test import *


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def run():
    opt = Config()

    if opt.display:
        visualizer = Visualizer()

    # device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FaceDataset(opt.train_root, opt.train_list,
                            phase='train', input_shape=opt.input_shape)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    print('{} train iters per epoch:'.format(len(trainloader)))

    # Focal Loss, 解决类别不均衡问题,减少易分类样本的权重，使得模型在训练时更专注于难分类的样本
    # https://blog.csdn.net/u014380165/article/details/77019084
    # 

    #定义损失函数
    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)  # 
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #定义模型
    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    #全连接层？
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(
            512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    #定义优化算法
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)

    # https://www.programcreek.com/python/example/98143/torch.optim.lr_scheduler.StepLR
    # ? 每过{lr_step}个epoch训练，学习率就乘gamma
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()  # train模式，eval模式
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            feature = model(data_input)
            output = metric_fc(feature, label)  # 全连接层？ 将原本用于输出分类的层，改成输出512维向量？似乎不是？
            loss = criterion(output, label) # criterion:做出判断的依据
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1) #最大值所在的索引？ index <-> one-hot相互转换
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(
                    time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(
                        iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(
                        iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        # train结束，模型设置为eval模式
        model.eval()

        #测试？
        identity_list = get_lfw_list(opt.lfw_test_list)
        img_paths = [os.path.join(opt.lfw_root, each)
                     for each in identity_list]
        acc = lfw_test(model, img_paths, identity_list,
                       opt.lfw_test_list, opt.test_batch_size)

        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')


if __name__ == '__main__':
    run()
