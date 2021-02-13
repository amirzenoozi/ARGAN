from tools.utils import load_obj, save_obj
from tools.mobile_net_layer import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout, depthwise_separable_conv2d_no_activation
import tensorflow as tf
import os
import numpy as np
import time

BGR_MEAN = [103.939, 116.779, 123.68]

class MobileNet:
    """
    MobileNet Class
    """

    def __init__(self, mobile_net_weight_path="mobilenet_weight/mobilenet_v1.pkl"):

        # init parameters and input
        self.nodes = {}
        self.pretrained_path = mobile_net_weight_path
        
        try:
            print("Loading ImageNet pretrained weights...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet_encoder')
            dict = load_obj( self.pretrained_path )
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def build(self, rgb):
        start_time = time.time()
        rgb_scaled = ((rgb + 1) / 2) * 255.0 # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - BGR_MEAN[0], green - BGR_MEAN[1], red - BGR_MEAN[2]])

        preprocessed_input = bgr

        # Model is here!
        conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * 1.0)), kernel_size=(3, 3), padding='SAME', stride=(2, 2), activation=tf.nn.relu6, batchnorm_enabled=True, is_training=False, l2_strength=4e-5, bias=0.0)
        # self.__add_to_nodes([conv1_1])
        
        ############################################################################################
        conv2_1_dw, conv2_1_pw = depthwise_separable_conv2d('conv_ds_2', conv1_1, width_multiplier=1.0,num_filters=64, kernel_size=(3, 3), padding='SAME',stride=(1, 1),batchnorm_enabled=True,activation=tf.nn.relu6,is_training=False,l2_strength=4e-5,biases=(0.0, 0.0))
        # self.__add_to_nodes([conv2_1_dw, conv2_1_pw])

        conv2_2_dw, conv2_2_pw = depthwise_separable_conv2d('conv_ds_3', conv2_1_pw, width_multiplier=1.0, num_filters=128, kernel_size=(3, 3), padding='SAME', stride=(2, 2), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))

        # Just For Calculate Loss
        self.no_activation_layer, self.a = depthwise_separable_conv2d_no_activation('conv_ds_3', conv2_1_pw, width_multiplier=1.0, num_filters=128, kernel_size=(3, 3), padding='SAME', stride=(2, 2), batchnorm_enabled=True, activation=None, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv2_2_dw, conv2_2_pw])
        
        ############################################################################################
        conv3_1_dw, conv3_1_pw = depthwise_separable_conv2d('conv_ds_4', conv2_2_pw, width_multiplier=1.0, num_filters=128, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv3_1_dw, conv3_1_pw])

        conv3_2_dw, conv3_2_pw = depthwise_separable_conv2d('conv_ds_5', conv3_1_pw, width_multiplier=1.0, num_filters=256, kernel_size=(3, 3), padding='SAME', stride=(2, 2), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv3_2_dw, conv3_2_pw])
        
        ############################################################################################
        conv4_1_dw, conv4_1_pw = depthwise_separable_conv2d('conv_ds_6', conv3_2_pw, width_multiplier=1.0, num_filters=256, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv4_1_dw, conv4_1_pw])

        conv4_2_dw, conv4_2_pw = depthwise_separable_conv2d('conv_ds_7', conv4_1_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(2, 2), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv4_2_dw, conv4_2_pw])
        
        ############################################################################################
        conv5_1_dw, conv5_1_pw = depthwise_separable_conv2d('conv_ds_8', conv4_2_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_1_dw, conv5_1_pw])

        conv5_2_dw, conv5_2_pw = depthwise_separable_conv2d('conv_ds_9', conv5_1_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_2_dw, conv5_2_pw])

        conv5_3_dw, conv5_3_pw = depthwise_separable_conv2d('conv_ds_10', conv5_2_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_3_dw, conv5_3_pw])

        conv5_4_dw, conv5_4_pw = depthwise_separable_conv2d('conv_ds_11', conv5_3_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_4_dw, conv5_4_pw])

        conv5_5_dw, conv5_5_pw = depthwise_separable_conv2d('conv_ds_12', conv5_4_pw, width_multiplier=1.0, num_filters=512, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_5_dw, conv5_5_pw])

        conv5_6_dw, conv5_6_pw = depthwise_separable_conv2d('conv_ds_13', conv5_5_pw, width_multiplier=1.0, num_filters=1024, kernel_size=(3, 3), padding='SAME', stride=(2, 2), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv5_6_dw, conv5_6_pw])
        
        ############################################################################################
        conv6_1_dw, conv6_1_pw = depthwise_separable_conv2d('conv_ds_14', conv5_6_pw, width_multiplier=1.0, num_filters=1024, kernel_size=(3, 3), padding='SAME', stride=(1, 1), batchnorm_enabled=True, activation=tf.nn.relu6, is_training=False, l2_strength=4e-5, biases=(0.0, 0.0))
        # self.__add_to_nodes([conv6_1_dw, conv6_1_pw])
        
        ############################################################################################
        avg_pool = avg_pool_2d(conv6_1_pw, size=(7, 7), stride=(1, 1))
        dropped = dropout(avg_pool, 0.999, tf.constant(False, dtype=tf.bool))
        # self.logits = flatten(conv2d('fc', dropped, kernel_size=(1, 1), num_filters=1001, l2_strength=4e-5, bias=0.0))
        # self.__add_to_nodes([avg_pool, dropped, self.logits])

        print(("build model finished: %fs" % (time.time() - start_time)))

    
    def __add_to_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.name] = node