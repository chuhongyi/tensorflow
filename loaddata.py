# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:47:18 2018

@author: 褚宏义
"""

import os
import numpy
from PIL import Image

class LoadData:
    
    def __init__(self):
        self.load_data('E:/flowerdatabase', 0)
        self.load_data('E:/testdatabase', 1)
        #self._split_train_valid(valid_rate = 0.8)
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
    '''   
    def _split_train_valid(self, valid_rate = 0.8):
        images, labels = self.train_images, self.train_labels 
        thresh = int(images.shape[0] * valid_rate)
        self.train_images, self.train_labels = images[0:thresh,:,:,:], labels[0:thresh]
        self.valid_images, self.valid_labels = images[thresh:,:,:,:], labels[thresh:]
    '''
    def load_data(self, path, id):
        #id=0读取训练集和验证集，不为0读取测试集
        # 读取训练集
        if(id == 0):
            train_images, train_labels = [], []
            valid_images, valid_labels = [], []
            filelist = os.listdir(path)
            for item in filelist:
                src = os.path.join(os.path.abspath(path), item)
                pics = os.listdir(src)
                for pic in pics:
                    img = Image.open(os.path.join(src, pic))
                    if pics.index(pic) < len(pics) * 0.8:
                        train_labels.append(item)
                        train_images.append(numpy.asarray(img, dtype = 'float'))
                    else:
                        valid_labels.append(item)
                        valid_images.append(numpy.asarray(img, dtype = 'float'))
            train_images = numpy.array(train_images, dtype = 'float')
            train_labels = numpy.array(train_labels, dtype = 'int')
            valid_images = numpy.array(valid_images, dtype = 'float')
            valid_labels = numpy.array(valid_labels, dtype = 'int')
            self.train_images, self.train_labels = train_images, train_labels
            self.valid_images, self.valid_labels = valid_images, valid_labels
        # 读取测试集
        else:
            test_images, test_labels = [], []
            filelist = os.listdir(path)
            for item in filelist:
                src = os.path.join(os.path.abspath(path), item)
                pics = os.listdir(src)
                for pic in pics: 
                    test_labels.append(item)
                    img = Image.open(os.path.join(src, pic))
                    test_images.append(numpy.asarray(img, dtype='float'))
            test_images = numpy.array(test_images, dtype = 'float')
            test_labels = numpy.array(test_labels, dtype = 'int')
            self.test_images, self.test_labels = test_images, test_labels