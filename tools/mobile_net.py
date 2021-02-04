from tensorflow.keras.layers import Input
import tensorflow.keras.applications as kerasApp
import tensorflow.keras.backend as K
import tensorflow as tf
import time


BGR_MEAN = [103.939, 116.779, 123.68]

class MobileNet:
    """
    MobileNet Class
    """
    def __init__(self):
        print('Model Loaded Successfully...')

    def build(self, rgb):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        # Preprocessing: Turning RGB to BGR - Mean.
        start_time = time.time()
        print( rgb )
        rgb_scaled = ((rgb + 1) / 2) * 255.0 # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        
        bgr = tf.concat(axis=3, values=[blue - BGR_MEAN[0], green - BGR_MEAN[1], red - BGR_MEAN[2]])
        # bgr = tf.image.resize_images(bgr, [224, 224])
        K.clear_session()
        model = kerasApp.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001, include_top=False, weights="imagenet", input_tensor=Input(shape=(256, 256, 3)), pooling=None, classes=1000)
        prediction = model.predict( bgr, steps=1 )
        self.no_activation_layer = model.get_layer('conv_dw_1_bn').output

        print(("build model finished: %fs" % (time.time() - start_time)))
