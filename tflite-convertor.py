from glob import glob
# from tqdm import tqdm

# import cv2
import os
import argparse

import tensorflow as tf

from tensorflow.python.platform import gfile


def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', type=str, default='Shinkai', help='dataset_name')

    return parser.parse_args()

def model_convertor():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")


    saved_tf_dir = "/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/AnimeGANv2.model-39"
    save_tflite_dir  = '/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/' + os.path.basename(saved_tf_dir) + '.tflite'





    sess = tf.Session()
    saver = tf.train.import_meta_graph(saved_tf_dir + ".meta")

    saver.restore(sess, saved_tf_dir)


    
    graph_ops = sess.graph.get_operations()
    ops_name  = [i.name for i in graph_ops]
    print( ops_name )
    

    # specify which tensor output you want to obtain 
    # (correspond to prediction result)
    your_outputs = ["generator/G_MODEL/out_layer/Conv/weights"]
    your_inputs = ["batch_normalization"]

    
    # in_tensors= [ your_inputs ]
    # out_tensors= [ your_outputs ]

    # # convert to tflite model
    # converter = tf.lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # ## Weight quantizations
    # #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    # tflite_model = converter.convert()
    # open(save_tflite_dir, "wb").write(tflite_model)
    # print("tflite model is successfully stored!")

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    model_convertor()
    # print( dir(tf.contrib.lite) )


if __name__ == '__main__':
    main()