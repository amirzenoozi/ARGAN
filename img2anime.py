'''
   made by @finnkso (github)
   2020.04.09
   tensorflow-gpu==1.15.0  : tf.compat.v1
   if tensorflow-gpu==1.8.0, please replayce tf.compat.v1 to tf
'''

from tools.utils import *
from tqdm import tqdm
from net import generator,generator_lite
from tools.utils import preprocessing, check_folder

import argparse
import os
import time
import cv2

import tkinter as tk
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_path', type=str, default='dataset/test/real/'+ '762.jpg', help='image file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1', help='Directory name to save the checkpoints')

    return parser.parse_args()

def convert_image(img, img_size):
    img = cv2.imread(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = np.asarray(img)
    return img

def run_model(img_path, checkpoint_dir, img_size=(256,256)):
    # gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    # if gpu_stat:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(allow_growth=True)
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.variable_scope("generator", reuse=False):
        if 'lite' in checkpoint_dir:
            test_generated = generator_lite.G_net(test_real).fake
        else:
            test_generated = generator.G_net(test_real).fake

    tfconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=tfconfig) as sess:
        # load model
        start_load_model_time = time.time()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [*] Failed to find a checkpoint")
            return
        
        # Load Model Time
        print(("Load Model Finished: %fs" % (time.time() - start_load_model_time)))

        start_style_transfer_time = time.time()
        sample_image = convert_image(img_path, img_size)
        test_real,test_generated = sess.run([test_real, test_generated],feed_dict = {test_real:sample_image} )
        save_images(test_generated, os.path.join( os.path.dirname(args.img_path), 'vid_name.png'), img_path)

        # Image Convert Duration
        print(("Load Model Finished: %fs" % (time.time() - start_style_transfer_time)))

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    run_model(args.img_path, args.checkpoint_dir)