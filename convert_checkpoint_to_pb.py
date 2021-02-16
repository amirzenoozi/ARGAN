"""
It's Just Work With Tensorflow v1.14.0
"""

import tensorflow as tf
import argparse
import re
import os


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--meta_file', type=str, default='AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/AnimeGANv2.model-39.meta', help='meta File Path')

    """checking arguments"""
    return parser.parse_args()

def main( arguments ):
    meta_path = 'checkpoint/' + arguments.meta_file  # Your .meta file
    output_node_names = ['generator/G_MODEL/out_layer/Tanh']    # Output nodes
    pb_file_name = re.search('\((.*?)\)', meta_path.split('/')[-2]).group(1)

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        restore_latest_checkpoint_path = os.path.abspath(os.path.join(meta_path, os.pardir))
        saver.restore(sess,tf.train.latest_checkpoint(restore_latest_checkpoint_path))

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open( f'/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/converted_models/pb/{pb_file_name}.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    print("Model Is Converted to PB File")

if __name__ == '__main__':
    arg = parse_args()
    main( arg )
