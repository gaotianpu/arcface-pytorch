#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019-03-13
@author: gaotianpu
"""
from __future__ import print_function
import os
import time
import cv2
import numpy as np
import torch
import torchvision
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from config.config import Config
from models.resnet import *


class FaceDataset(Dataset):
    """face dataset
    参考： https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, data_file, root_dir, transform=None):
        self.data_file = data_file
        self.root_dir = root_dir
        self.transform = transform

        self.pairs_list = []
        self.pairs_len = 0
        self.parse_data_file()

        assert self.pairs_list

    def __len__(self):
        return self.pairs_len

    def __getitem__(self, idx):
        pair_row = self.pairs_list[idx]

        # 图片0, 图片1，是否一个人？
        # return self.load_image(img_path)
        item = {"img0": self.load_image(pair_row[0]),
                "img1": self.load_image(pair_row[1]),
                "label": float(pair_row[2])}

        if self.transform:
            item = self.transform(item)

        return item

    def parse_data_file(self):
        with open(self.data_file, 'r') as fd:
            count = -1
            for count, line in enumerate(fd):
                self.pairs_list.append(line.strip().split()) #数据量巨大的情况下，不适合载入到内存？
            count += 1
            self.pairs_len = count

            # pairs = fd.readlines()
            # self.pairs_list = [pair.strip().split() for pair in pairs]
            # self.pairs_len = len(self.pairs_list)

    def load_image(self, img_file):
        """读取图像信息"""
        img_path = os.path.join(self.root_dir, img_file)

        # image = io.imread(img_path)
        # print("image.shape", image.shape) # (128, 128, 3)

        # Load an color image in grayscale
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
        image = cv2.imread(img_path, 0)
        if image is None:
            return None

        # print( image.shape ) # (128, 128)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
        # Flip array in the left/right direction.
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.fliplr.html python矩阵水平镜像

        image = np.dstack((image, np.fliplr(image)))  # 相当于将图片进行水平翻转
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


### ~~~~

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
        else:
            images = np.concatenate((images, image), axis=0)
        # print(images.shape) # (n, 1, 128, 128)
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            print("data.shape:", data.shape)  # torch.Size([60, 1, 128, 128])
            # data = data.to(torch.device("cuda"))
            output = model(data)  # 获得512维的向量
            output = output.data.cpu().numpy()  # ？cpu
            # print(output.shape) # (60, 512)

            # https://stackoverflow.com/questions/7123888/what-is-double-colon-in-numpy-like-in-myarray03
            #原始图像和镜像图像一起打包放入feature中
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


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dictp


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


def test():
    opt = Config()
    face_data = FaceDataset(opt.lfw_test_list, opt.lfw_root)
    img0 = face_data[0]['img0']
    img1 = face_data[0]['img1']
    label = face_data[0]['label']

    images = img0
    images = np.concatenate((images, img1), axis=0)

    data = torch.from_numpy(images)
    # print(images.shape) # (4, 1, 128, 128)

    print(img0.shape)
    print(img0)


# ~~~~

def load_test():
    opt = Config()
    face_data = FaceDataset(opt.lfw_test_list, opt.lfw_root)
    print(face_data[0]['img1'].shape)

    data_loader = DataLoader(face_data,
                             batch_size=4,
                             shuffle=False,
                             num_workers=1)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched['img0'].shape,sample_batched['img1'].shape,sample_batched['label'].shape)
        if i_batch>1:
            break 


if __name__ == '__main__':
    load_test()
    # test()

    # opt = Config()
    # if opt.backbone == 'resnet18':
    #     model = resnet_face18(opt.use_se)
    # elif opt.backbone == 'resnet34':
    #     model = resnet34()
    # elif opt.backbone == 'resnet50':
    #     model = resnet50()

    # # You can easily run your operations on multiple GPUs by making your model run parallelly using DataParallel
    # model = DataParallel(model)
    # # load_model(model, opt.test_model_path)
    # model.load_state_dict(torch.load(opt.test_model_path, map_location='cpu' ))
    # # model.to(torch.device("cuda"))

    # #获取所有图像的相对路径
    # identity_list = get_lfw_list(opt.lfw_test_list)
    # #获取图像的绝对路径
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    # model.eval()
    # lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
