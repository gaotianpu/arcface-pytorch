#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import time
import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from config import Config
from models.resnet import *


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def load_image(img_path):
    # Load an color image in grayscale
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
    # Flip array in the left/right direction. 
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.fliplr.html python矩阵水平镜像
    # print( image.shape ) # (128, 128)

    # np.fliplr(image) #左右对称变换
    image = np.dstack((image, np.fliplr(image)))

    # print( image.shape ) # (128, 128, 2)

    image = image.transpose((2, 0, 1))
    # print( image.shape ) # (2, 128, 128)
    image = image[:, np.newaxis, :, :]
    # print( image.shape ) # (2, 1, 128, 128)
    # print( image) #int
    image = image.astype(np.float32, copy=False)
    image -= 127.5  # 将0~255的值转成 -127.5 ~ 127.5 ？
    # print( image)
    image /= 127.5  # 再归一处理 转换为-1，1之间的float数值
    # print( image.shape ) # (2, 1, 128, 128)
    # print( image) #归一？
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        print(img_path)
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
            print('images none shape', images.shape)
        else:
            images = np.concatenate((images, image), axis=0)
        print("images.shape:", images.shape)  # (n, 1, 128, 128)
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            # print("data.shape:",data.shape)  # torch.Size([60, 1, 128, 128])
            # data = data.to(torch.device("cuda"))
            output = model(data)  # 获得512维的向量
            output = output.data.cpu().numpy()  # ？cpu
            # print("output.shape:" , output.shape) # (60, 512)

            # https://stackoverflow.com/questions/7123888/what-is-double-colon-in-numpy-like-in-myarray03
            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape) # (30, 1024)
            # break

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))
            # print ( features.shape) # (30, 1024)
            # break
            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()

        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    # 全部装载至内存中 ？
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    # print(features.shape)
    t = time.time() - s

    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)

    #key= Abel_Pacheco/Abel_Pacheco_0001.jpg
    #value = 向量
    # for k,v in fe_dict.items():
    #     print('fe_dict')
    #     print(k,v)
    #     break
    # return

    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':
    opt = Config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #获取所有图像的相对路径
    identity_list = get_lfw_list(opt.lfw_test_list)
    #获取图像的绝对路径
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se) #
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50() 

    # You can easily run your operations on multiple GPUs by making your model run parallelly using DataParallel
    # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path, map_location=device)) 
    model.to(device)  
    model.eval()

    lfw_test(model, img_paths, identity_list,
             opt.lfw_test_list, opt.test_batch_size)
