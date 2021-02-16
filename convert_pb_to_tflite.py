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

    parser.add_argument('--pb_file', type=str, default='output_graph.pb', help='pb File Path')

    """checking arguments"""
    return parser.parse_args()

def main( arguments ):
    graph_def_file = 'converted_models/pb/' + arguments.pb_file
    base_file = os.path.basename( graph_def_file ).replace('.pb', '')

    input_arrays = ["generator/G_MODEL/A/MirrorPad"]
    output_arrays = ["generator/G_MODEL/out_layer/Tanh"]

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=graph_def_file,
        input_arrays=input_arrays,
        input_shapes={'generator/G_MODEL/A/MirrorPad' : [5, 262, 262,3]},
        output_arrays=output_arrays
    )
    tflite_model = converter.convert()

    # # Save the model.
    with open( f'converted_models/tflite/{base_file}.tflite', 'wb') as f:
      f.write(tflite_model)

    print("Model Is Converted to Tf-Lite Version")
  


if __name__ == '__main__':
    arg = parse_args()
    main( arg )