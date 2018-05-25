# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:04:11 2018

@author: 褚宏义
"""

from skimage import io, transform
import tensorflow as tf
import numpy

flower_dict = {'0': '百合', '1': '玫瑰', '2': '茉莉花', '3': '荷花', '4': '昙花', '5': '桃花'}
path = 'F:/1.jpg'

image_size, n_channel = 64, 3#图片尺寸：64*64*3

#读入图片，改变尺寸
def read_image(path):
    img = io.imread(path)
    img = transform.resize(img, (image_size, image_size))
    return numpy.asarray(img, dtype = 'float32')

data = []
data.append(read_image(path))

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model3/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model3/'))#导入变量值
    
    graph = tf.get_default_graph()#导入图
    
    input_image = graph.get_tensor_by_name('images: 0')
    keep_prob = graph.get_tensor_by_name('keep_prob: 0')
    logit = graph.get_operation_by_name('logits: 0')
    classification_result = sess.run(logit, feed_dict = {input_image: data, keep_prob: 1.})
    prob = tf.nn.softmax(classification_result).eval()
    output = tf.argmax(prob, 1).eval()
    print(output)
    #print(graph.collections)