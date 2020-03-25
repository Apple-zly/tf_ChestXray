#!/usr/bin/env python
# coding:utf-8
"""
Name    : Vgg.py
Author  : lijun.chen
Github  : github.com/kiclent
Contect : lijun.chen@tongdun.net
Time    : 2020-02-15 12:12
Desc    :
"""
import sys
sys.path.append('../')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
VGG_MEAN = [103.939, 116.779, 123.68]
def get_batch(data_dir, csv_data, batch_size):
	n = len(csv_data)
	idx = np.random.permutation(n)

	batch_i = 0


	while batch_i < n:
		xs = []
		ys = []
		_idx = idx[batch_i:batch_i+batch_size]
		for i in range(len(_idx)):
			_img = np.asarray(cv2.imread(os.path.join(data_dir, csv_data[_idx[i]][0])), np.float32)
			#标准化255
			_img[..., 0] = _img[..., 0] - VGG_MEAN[2]
			_img[..., 1] = _img[..., 1] - VGG_MEAN[1]
			_img[..., 2] = _img[..., 2] - VGG_MEAN[0]
			xs.append(_img)

			_y = np.zeros((4, ), np.float32)
			_y[int(csv_data[_idx[i]][1])] = 1.0
			ys.append(_y)
		yield np.asarray(xs), np.asarray(ys)
		batch_i += batch_size




# 数据读取
csv_file_train = '/home/data/ChestXray/dataset/train/train.csv'
train_data_dir = '/home/data/ChestXray/dataset/train/'
csv_data_train = np.array(pd.read_csv(os.path.join(csv_file_train)))


csv_file_test = '/home/data/ChestXray/dataset/test/test.csv'
test_data_dir = '/home/data/ChestXray/dataset/test/'
csv_data_test = np.array(pd.read_csv(os.path.join(csv_file_test)))


#
H, W, C = 224, 224, 3
num_class = 4
model_name = 'Vgg16'


total_epochs = 50
batch_size = 32
weight_decay = 1e-4
init_learning_rate = 1e-1
init_drop_rate = 0.2
reduce_lr_epoch = [25, 37]
epoch_learning_rate = init_learning_rate


# ------------------------------------
tf_X = tf.placeholder(tf.float32, shape=[None, H, W, C], name='tf_X')
tf_Y = tf.placeholder(tf.float32, shape=[None, num_class], name='tf_Y')
training_flag = tf.placeholder(tf.bool, name='training_flag')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

from lib.models_vgg import Vgg16

# logits = DenseNet_BC_SE(
#     x=tf_X,
#     num_class=num_class,
#     nb_layers=(6, 12, 24, 16),
#     growth_k=16,
#     ratio=4,
#     theta=0.5,
#     training=training_flag,
#     dropout_rate=dropout_rate,
#     name=model_name
# ).model

# prediction = Softmax(logits)
# prediction = Vgg16.build_model(tf_X)
prediction = Vgg16(tf_X, name=model_name).output


# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_Y, logits=prediction))
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
loss = cross_entropy + weight_decay*l2_loss

# 优化器
SGD = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
optimizer = SGD.minimize(loss)

# 准确率计算
correct_prediction = tf.equal(tf.argmax(tf_Y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 验证
def eva(sess):

	test_acc, test_loss = 0, 0
	for batch_xs, batch_ys in get_batch(test_data_dir, csv_data_test, batch_size):

		test_feed_dict = {
			tf_X: batch_xs,
			tf_Y: batch_ys,
			dropout_rate: init_drop_rate,
			training_flag: False
		}
		_loss, _acc = sess.run([loss, accuracy], feed_dict=test_feed_dict)

		w = batch_xs.shape[0] / csv_data_test.shape[0]
		test_acc += _acc * w
		test_loss += _loss * w

	return test_acc, test_loss


# 训练
def train_epoch(sess, lr):

	train_acc, train_loss = 0, 0
	step = 0
	for batch_xs, batch_ys in get_batch(train_data_dir, csv_data_train, batch_size):
		step += 1
		train_feed_dict = {
			tf_X: batch_xs,
			tf_Y: batch_ys,
			learning_rate: lr,
			dropout_rate: init_drop_rate,
			training_flag: True
		}
		# print(batch_xs.shape, batch_ys.shape)
		_, _loss, _acc = sess.run([optimizer, loss, accuracy], feed_dict=train_feed_dict)

		print("step= {} | loss= {:.5f} | acc= {:.5f}".format(step, _loss, _acc))
		w = batch_xs.shape[0] / csv_data_train.shape[0]
		train_acc += _acc * w
		train_loss += _loss * w
	return train_acc, train_loss


from time import time as tc
history = []

with tf.Session() as sess:


	saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
	sess.run(init)

	# tensorboard 可视化
	merged = tf.summary.merge_all()
	summary_writer_train = tf.summary.FileWriter('logs/train', sess.graph)
	summary_writer_test = tf.summary.FileWriter('logs/test')


	for epoch in range(1, total_epochs+1):
		if epoch in reduce_lr_epoch:
			epoch_learning_rate /= 10

		# 训练
		tic = tc()
		train_acc, train_loss = train_epoch(sess, epoch_learning_rate)
		train_time = tc() - tic

		# 测试
		test_acc, test_loss = eva(sess)

		print('\nepoch: {}, training time: {:.0f} s.'.format(epoch, train_time))
		print('train loss: {:.5f}, accuracy: {:.3f}'.format(train_loss, train_acc))
		print('test  loss: {:.5f}, accuracy: {:.3f}'.format(test_loss, test_acc))

		history.append([epoch, train_acc, train_loss, test_acc, test_loss, tc() - tic])

		train_summary = tf.Summary()
		train_summary.value.add(tag='000/accuracy', simple_value=train_acc)
		train_summary.value.add(tag='000/loss', simple_value=train_loss)
		summary_writer_train.add_summary(train_summary, global_step=epoch)

		test_summary = tf.Summary()
		test_summary.value.add(tag='000/accuracy', simple_value=test_acc)
		test_summary.value.add(tag='000/loss', simple_value=test_loss)
		summary_writer_test.add_summary(test_summary, global_step=epoch)

	# 保存模型
	saver.save(sess, './model_saves/Vgg.ckpt')

import numpy as np
history = np.array(history)

print('\n------------------- Fashion-mnist -------------------')
print('Best testing accuracy: {:.3f}'.format(history[:, 3].max()))
print('Training time: {:.0f} s.'.format(history[:, -1].sum()))

# 保存结果
import pandas as pd
pd.DataFrame(
	history,
	columns=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'time']
).to_csv('history.csv', index=None, encoding='utf-8')

if __name__ == '__main__':
    pass

