# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:36:52 2018

@author: 褚宏义
"""

import sys
import os
import tensorflow as tf
import numpy
import time
from conv_layer import ConvLayer
from pool_layer import PoolLayer
from dense_layer import DenseLayer
from cifar10 import Corpus
from loaddata import LoadData

class ConvNet():
    def __init__(self, n_channel = 3, n_classes = 2, image_size = 64):
        self.images = tf.placeholder(dtype = tf.float32, shape = [None, image_size, image_size, n_channel], name = 'images')
        self.labels = tf.placeholder(dtype = tf.int64, shape= [None], name = 'labels')
        self.keep_prob = tf.placeholder(dtype = tf.float32, name = 'keep_prob')
        self.global_step = tf.Variable(0, dtype = tf.int32, name = 'global_step')
        
        #cnn网络
        conv_layer1 = ConvLayer(input_shape = (None, image_size, image_size, n_channel),
            n_size = 3,#卷积核3*3
            n_filter = 64, 
            stride = 1,
            activation = 'relu',
            batch_normal = True,
            weight_decay = 1e-4,
            name = 'conv1')
        
        pool_layer1 = PoolLayer(n_size = 2, stride = 2, mode = 'max', resp_normal = True, name = 'pool1')
        
        conv_layer2 = ConvLayer(input_shape=(None, int(image_size / 2), int(image_size / 2), 64),
            n_size = 3,
            n_filter = 128,
            stride = 1,
            activation = 'relu',
            batch_normal = True,
            weight_decay = 1e-4,
            name = 'conv2')
        pool_layer2 = PoolLayer(n_size = 2, stride = 2, mode = 'max', resp_normal = True, name = 'pool2')
        
        conv_layer3 = ConvLayer(input_shape = (None, int(image_size / 4), int(image_size / 4), 128),
            n_size = 3,
            n_filter = 256, 
            stride = 1,
            activation = 'relu',
            batch_normal = True,
            weight_decay = 1e-4, 
            name = 'conv3')
        pool_layer3 = PoolLayer(n_size = 2, stride = 2, mode = 'max', resp_normal = True, name = 'pool3')
        
        dense_layer1 = DenseLayer(input_shape = (None, int(image_size / 8) * int(image_size / 8) * 256),
            hidden_dim = 1024, 
            activation = 'relu',
            dropout = True,
            keep_prob = self.keep_prob, 
            batch_normal = True,
            weight_decay = 1e-4,
            name = 'dense1')
        
        dense_layer2 = DenseLayer(input_shape = (None, 1024),
            hidden_dim = n_classes,
            activation = 'none',
            dropout = False,
            keep_prob = None, 
            batch_normal = False,
            weight_decay = 1e-4,
            name = 'dense2')
        
        #数据流
        hidden_conv1 = conv_layer1.get_output(input = self.images)
        hidden_pool1 = pool_layer1.get_output(input = hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input = hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input = hidden_conv2)
        hidden_conv3 = conv_layer3.get_output(input = hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input = hidden_conv3)
        input_dense1 = tf.reshape(hidden_pool3, [-1, int(image_size / 8) * int(image_size / 8) * 256])
        output_dense1 = dense_layer1.get_output(input = input_dense1)
        logits = dense_layer2.get_output(input = output_dense1)
        logits = tf.multiply(logits, 1, name = 'logits')
        
        # 目标/损失函数
        self.objective = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 500), 
            lambda: tf.constant(0.01),
            lambda: tf.cond(tf.less(self.global_step, 1000), 
                lambda: tf.constant(0.001),
                lambda: tf.constant(0.0001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.avg_loss, global_step = self.global_step)
        
        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
    def train(self, dataloader, backup_path, n_epoch = 10, batch_size = 30):
        best_train_loss = float('inf')
        start_time = time.clock()
        #获取输入数据
        train_images = dataloader[0][0]
        train_labels = dataloader[0][1]
        valid_images = dataloader[1][0]
        valid_labels = dataloader[1][1]
        #构建session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.45)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        #模型保存器
        self.saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 1)
        #模型初试化
        self.sess.run(tf.global_variables_initializer())
        #模型训练
        for epoch in range(n_epoch):
            
            """
            #数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode = 'train', flip = True, crop = True, crop_shape = (24, 24, 3), whiten = True, noise = False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode = 'test', flip = False, crop = True, crop_shape = (24, 24, 3), whiten = True, noise = False)
            valid_labels = dataloader.calid_labels
            """
            
            #开始本轮训练，并计算目标函数值
            train_loss = 0.0
            for i in range(0, data.n_train, batch_size):
                batch_images = train_images[i: i + batch_size]
                batch_labels = train_labels[i: i + batch_size]
                [_, avg_loss, iteration] = self.sess.run(fetches = [self.optimizer, self.avg_loss, self.global_step],
                    feed_dict = {self.images: batch_images,
                        self.labels: batch_labels,
                        self.keep_prob: 0.5})
                train_loss += avg_loss * batch_images.shape[0]
            train_loss = 1.0 * train_loss / data.n_train
            
            #训练后获取本轮验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, data.n_valid, batch_size):
                batch_images = valid_images[i: i + batch_size]
                batch_labels = valid_labels[i: i + batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(fetches = [self.accuracy, self.avg_loss],
                    feed_dict = {self.images: batch_images,
                        self.labels: batch_labels,
                        self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss / batch_images.shape[0]
                
            valid_accuracy = 1.0 * valid_accuracy / data.n_valid
            valid_loss = 1.0 * valid_loss / data.n_valid
            print('epoch{%d}, iter[%d], train loss: %.6f, valid loss: %.6f, valid accuracy: %.4f' % (epoch, iteration, train_loss, valid_loss, valid_accuracy * 100) + '%')
            sys.stdout.flush()
                
            #保存模型
            #当loss最小时
            if best_train_loss >= train_loss:
                best_train_loss = train_loss
                self.saver.save(self.sess, os.path.join(backup_path, 'model.ckpt'))
                print('model has been updated at epoch{%d} ,iter[%d] with train_loss: %.6f, valid_loss: %.6f and valid_accuracy: %.4f' % (epoch, iteration, best_train_loss, valid_loss, 100 * valid_accuracy) + '%')
        self.sess.close()
        end_time = time.clock()
        print("ran for %.2f minutes" % ((end_time - start_time) / 60.))
           
    def test(self, dataloader, backup_path, batch_size = 30):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.25)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        #读取模型
        self.saver = tf.train.Saver()
        model_path = os.path.join(backup_path, 'model.ckpt')
        assert(os.path.exists(model_path + '.index'))
        self.saver.restore(self.sess, model_path)
        print('Successfully read model from %s' % (model_path))
        #在测试集上计算准确率
        accuracy_list = []
        test_images = dataloader[0]
        test_labels = dataloader[1]
        #test_images = dataloader.data_augmenttation(dataloader.test_images, flip = False, crop = True, crop_shape = (24, 24, 3), whiten = True, noise = False)
        #test_labels = dataloader.test_labels
        for i in range(0, data.n_test, batch_size):
            batch_images = test_images[i: i + batch_size]
            batch_labels = test_labels[i: i + batch_size]
            [avg_accuracy] = self.sess.run(fetches = [self.accuracy],
            feed_dict = {self.images: batch_images,
                         self.labels: batch_labels,
                         self.keep_prob: 1.0})
            accuracy_list.append(avg_accuracy)
        print('test precision: %.4f' % (100 * numpy.mean(accuracy_list)) + '%')
        self.sess.close()

cnn = ConvNet()
data = LoadData()
#data = Corpus()
cnn.train(([data.train_images, data.train_labels], [data.valid_images, data.valid_labels]), 'model3/')
cnn.test([data.test_images, data.test_labels], 'model3/')