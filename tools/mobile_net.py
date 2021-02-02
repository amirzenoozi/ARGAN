import tensorflow.contrib as tc
import tensorflow as tf
import numpy as np
import math
import time
import sys
import os
import pickle


BGR_MEAN = [103.939, 116.779, 123.68]

class MobileNet:
    """
    MobileNet Class
    """
    def __init__(self, input_size=224, classnum=6):
        self.input_size = input_size
        self.classnum = classnum
        self.normalizer = tc.layers.batch_norm


    def _create_placeholders(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.classnum], name="input_y")
        self.is_training = tf.placeholder(tf.bool)
        self.bn_params = {'is_training': self.is_training, 'scope': 'BatchNorm', 'scale': True}

    def _build_model(self, rgb):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        # Preprocessing: Turning RGB to BGR - Mean.
        start_time = time.time()
        rgb_scaled = ((rgb + 1) / 2) * 255.0 # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # print( red, green, blue)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        
        bgr = tf.concat(axis=3, values=[blue - BGR_MEAN[0], green - BGR_MEAN[1], red - BGR_MEAN[2]])

        i = 0
        # self.conv1 = tc.layers.conv2d(self.input, num_outputs=32, kernel_size=3, stride=2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}'.format(i))
        self.conv1 = tc.layers.conv2d(bgr, num_outputs=32, kernel_size=3, stride=2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}'.format(i))

        # 1
        i += 1
        self.dconv1 = tc.layers.separable_conv2d(self.conv1, num_outputs=None, kernel_size=3, depth_multiplier=1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv1 = tc.layers.conv2d(self.dconv1, 64, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 2
        i += 1
        self.dconv2 = tc.layers.separable_conv2d(self.pconv1, None, 3, 1, 2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv2 = tc.layers.conv2d(self.dconv2, 128, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 3
        i += 1
        self.dconv3 = tc.layers.separable_conv2d(self.pconv2, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv3 = tc.layers.conv2d(self.dconv3, 128, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 4
        i += 1
        self.dconv4 = tc.layers.separable_conv2d(self.pconv3, None, 3, 1, 2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv4 = tc.layers.conv2d(self.dconv4, 256, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 5
        i += 1
        self.dconv5 = tc.layers.separable_conv2d(self.pconv4, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv5 = tc.layers.conv2d(self.dconv5, 256, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 6
        i += 1
        self.dconv6 = tc.layers.separable_conv2d(self.pconv5, None, 3, 1, 2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv6 = tc.layers.conv2d(self.dconv6, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_1
        i += 1
        self.dconv71 = tc.layers.separable_conv2d(self.pconv6, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv71 = tc.layers.conv2d(self.dconv71, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_2

        i += 1
        self.dconv72 = tc.layers.separable_conv2d(self.pconv71, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv72 = tc.layers.conv2d(self.dconv72, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_3
        i += 1
        self.dconv73 = tc.layers.separable_conv2d(self.pconv72, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv73 = tc.layers.conv2d(self.dconv73, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_4
        i += 1
        self.dconv74 = tc.layers.separable_conv2d(self.pconv73, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv74 = tc.layers.conv2d(self.dconv74, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_5
        i += 1
        self.dconv75 = tc.layers.separable_conv2d(self.pconv74, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv75 = tc.layers.conv2d(self.dconv75, 512, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 8
        i += 1
        self.dconv8 = tc.layers.separable_conv2d(self.pconv75, None, 3, 1, 2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv8 = tc.layers.conv2d(self.dconv8, 1024, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 9
        i += 1
        self.dconv9 = tc.layers.separable_conv2d(self.pconv8, None, 3, 1, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv9 = tc.layers.conv2d(self.dconv9, 1024, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        with tf.variable_scope('global_avg_pooling'):
            self.pool = tc.layers.avg_pool2d(self.pconv9, kernel_size=7, stride=1)
        with tf.variable_scope('Logits'):
            self.output = tc.layers.conv2d(self.pool, self.classnum, 1, activation_fn=None, scope='Conv2d_1c_1x1')
            shapes = self.output.get_shape().as_list()
            self.out = tf.reshape(self.output, [-1, shapes[1] * shapes[2] * shapes[3]])

        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        
        print(("build model finished: %fs" % (time.time() - start_time)))