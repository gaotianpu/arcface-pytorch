#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  

def generate_train_meta():
    root_dir = 'lfw-align-128'
    names = list(os.listdir(root_dir))
    names.sort()
    # print(name_list.index('Alexandre_Despatie'))
    for i,name in enumerate(names):
        name_idx = names.index(name)
        img_dir = os.path.join(root_dir, name)
        for ii,img in enumerate(os.listdir(img_dir)):
            img_path = os.path.join(name, img)
            print('\t'.join([img_path,str(name_idx)]))
    
    # python make_train_meta.py > data/train.txt
    # sort -R data/train.txt > data/train.txt.rand
    # head -10000 train.txt.rand  > train_list.txt
    # tail -3233 train.txt.rand > val_list.txt


def generate_test_meta():
    pass 

if __name__ == "__main__":
    generate_train_meta()
    

       
