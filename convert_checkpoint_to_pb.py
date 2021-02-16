"""
It's Just Work With Tensorflow v1.14.0
"""

import tensorflow as tf

meta_path = '/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/AnimeGANv2.model-39.meta' # Your .meta file
output_node_names = ['generator/G_MODEL/out_layer/Tanh']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

print("Model Is Converted to PB File")
