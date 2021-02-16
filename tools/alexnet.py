import tensorflow as tf
import numpy as np
import sys
import time

VGG_MEAN = [103.939, 116.779, 123.68]

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, weights_path='weight/bvlc_alexnet.npy'):


        if weights_path is not None:
            self.data_dict = np.load(weights_path, encoding='bytes').item()
            print("npy file loaded ------- ",weights_path)
        else:
            self.data_dict = None
            print("npy file load error!")
            sys.exit(1)

        # Parse input arguments into class variables
        self.NUM_CLASSES = 1000
        self.KEEP_PROB = tf.constant(1.0)
        self.SKIP_LAYER = []

    def create(self, rgb):

        start_time = time.time()
        rgb_scaled = ((rgb + 1) / 2) * 255.0 # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])

        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = self.conv( bgr, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = self.lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = self.max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = self.conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = self.lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = self.max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = self.conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = self.fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = self.fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

        print(("build model finished: %fs" % (time.time() - start_time)))

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        """Create a convolution layer.
        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


    def fc(self, x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)


    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


    def dropout(self, x, keep_prob):
        """Create a dropout layer."""
        return tf.nn.dropout(x, keep_prob)