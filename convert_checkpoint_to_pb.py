import tensorflow as tf

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# graph_def_file = "/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/b1.pb"
# input_arrays = ["batch_normalization"]
# output_arrays = ["generator/G_MODEL/out_layer/Conv/weights"]

# # /media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/AnimeGANv2.model-39

# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
#   graph_def_file, input_arrays, output_arrays, input_shapes= {"main_input" : [1, 3, 256, 256]})
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #converter.experimental_new_converter = True
# #converter.allow_custom_ops = 1
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# ## Weight quantizations
# #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# #converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
# converter.target_spec.supported_types = [tf.float16]
# tflite_model = converter.convert()
# open("/home/sensifai/mehrdad/models/tagging/converted_by_first_method/b2.tflite", "wb").write(tflite_model)
# print("model is converted")

meta_path = '/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/AnimeGANv2.model-39.meta' # Your .meta file
output_node_names = ['generator/G_MODEL/out_layer/']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1_(res18_block1_2)/'))

    # Freeze the graph
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('/media/amirzenoozi/500G/Amirhossein/University/AnimeGANv2/output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
