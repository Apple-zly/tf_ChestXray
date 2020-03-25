'''
2019-01-17 14:13:05
author: kiclent@yahoo.com
Tensorflow 1.7.1
'''

from lib.tf_ops import *

class DenseNet_BC_SE():

    def __init__(self,
                 x,
                 training,
                 num_class=2,
                 growth_k=12,  # growth_k
                 ratio=4,
                 theta=0.5,
                 nb_layers=(6, 12, 24, 16),
                 dropout_rate=0,
                 name='DenseNet'):
        self.training = training
        self.growth_k = growth_k
        self.nb_layers = nb_layers
        self.ratio = ratio
        self.theta = theta
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.name = name

        self.model = self.build_model(x)


    def build_model(self, input_x):

        x = conv_layer(input_x, filter=2*self.growth_k, kernel=[3, 3], stride=2, layer_name=self.name+'conv0')
        x = Max_pooling(x, (3, 3))

        for i in range(len(self.nb_layers) - 1) :
            print('\n================= dense block {} ================='.format(i))
            x = self.dense_block(input_x=x, nb_layers=self.nb_layers[i], layer_name=self.name+'dense_block_'+str(i))
            x = self.transition_layer(x, scope=self.name+'trans_'+str(i))

        print('\n================= dense block {} ================='.format('final'))
        x = self.dense_block(input_x=x, nb_layers=self.nb_layers[-1], layer_name=self.name+'dense_block_final')
        x = Batch_Normalization(x, training=self.training, scope=self.name+'linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = Flatten(x)
        print(x)

        x = Fully_connected(x, self.num_class, layer_name=self.name+'FC_final_{}'.format(self.num_class))
        return x

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.growth_k, kernel=[1, 1], layer_name=scope+'_conv1')
            x = Dropout(x, rate=self.dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.growth_k, kernel=[3, 3], layer_name=scope+'_conv2')
            x = Dropout(x, rate=self.dropout_rate, training=self.training)

            print(x)
            return x

    def transition_layer(self, x, scope):
        in_channel = x.get_shape().as_list()[-1]
        out_channel = round(in_channel*self.theta)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=out_channel, kernel=[1,1], layer_name=scope+'_conv1')
            x = Dropout(x, rate=self.dropout_rate, training=self.training)
            x = Avg_pooling(x, pool_size=[2, 2], stride=2)
            print(x)

            return x

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim // ratio, layer_name=layer_name + '_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale


    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            x = self.squeeze_excitation_layer(x, x.get_shape().as_list()[-1], ratio=self.ratio,
                                              layer_name=layer_name + '_SE' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                x = self.squeeze_excitation_layer(x, x.get_shape().as_list()[-1], ratio=self.ratio,
                                                  layer_name=layer_name + '_SE' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)
            print(x)
            return x


class Resnet():
    pass

