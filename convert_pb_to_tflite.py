"""
It's Just Work With Tensorflow v1.14.0
"""

import tensorflow as tf


graph_def_file = "/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/output_graph.pb"


input_arrays = ["generator/G_MODEL/A/MirrorPad"]
output_arrays = ["generator/G_MODEL/out_layer/Tanh"]

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=graph_def_file,
    input_arrays=input_arrays,
    input_shapes={'generator/G_MODEL/A/MirrorPad' : [5, 262, 262,3]},
    output_arrays=output_arrays
)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model Is Converted to Tf-Lite Version")