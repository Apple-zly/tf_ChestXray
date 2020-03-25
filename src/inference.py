#!/usr/bin/env python
# coding:utf-8
"""
Name    : inference.py
Author  : .mat
Github  : github.com/kiclent
Contect : kiclent@yahoo.com
Time    : 2020-02-14 11:16
Desc    :
"""

import sys
sys.path.append('../')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import pdb

#
batch_size = 128
#model_name = 'DenseNet-k24'
model_name = 'Vgg16'
ckpt_path = './model_saves/Vgg.ckpt'

data_dir = '/home/data/ChestXray/dataset/evaluate/'
csv_data = csv_data_train = np.array(pd.read_csv(os.path.join(data_dir, 'upload.csv')))

VGG_MEAN = [103.939, 116.779, 123.68]

graph = tf.get_default_graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(ckpt_path+'.meta')
    tf_X = graph.get_tensor_by_name('tf_X:0')
    tf_Y = graph.get_tensor_by_name('tf_Y:0')
    training_flag = graph.get_tensor_by_name('training_flag:0')
    learning_rate = graph.get_tensor_by_name('learning_rate:0')
    dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
    #prediction = graph.get_tensor_by_name('Softmax:0')
    prediction = graph.get_tensor_by_name('out:0')


with tf.Session() as sess:

    saver.restore(sess, ckpt_path)


    preds = []
    n = len(csv_data)
    batch_i = 0
    while batch_i < n:
        xs = []
        for fn in csv_data[batch_i:batch_i+batch_size, 0]:
            #print(csv_data[batch_i:batch_i+batch_size, 0])
            #print('\n')
            #print(fn)
            _img = np.asarray(cv2.imread(os.path.join(data_dir, fn)), np.float32)
            _img[..., 0] = _img[..., 0] - VGG_MEAN[2]
            _img[..., 1] = _img[..., 1] - VGG_MEAN[1]
            _img[..., 2] = _img[..., 2] - VGG_MEAN[0]
            xs.append(_img)
            #print(_img.shape)

            batch_i += 1
        
        xs = np.asarray(xs)
        #print(xs.shape)
        test_feed_dict = {
            tf_X: xs,
            training_flag: False,
            dropout_rate: 0.2
        }
        probs = sess.run(prediction, feed_dict=test_feed_dict)
        print(probs)
        preds.append(np.argmax(probs, axis=1).reshape(-1))

        print('{} / {}'.format(batch_i, n))

    # 预测结束, 将每个Batch的识别结果拼接
    preds = np.hstack(preds)

    submit = pd.read_csv(os.path.join(data_dir, 'upload.csv'))
    submit['labels'] = preds

    submit.to_csv('submit_vgg.csv', index=False)








