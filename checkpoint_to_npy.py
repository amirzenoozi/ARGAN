from tensorflow.python import pywrap_tensorflow
import numpy as np
import os

# Create a dictionary, store name and array layers, each layer of the structure related to the name of the model
# autoencoder = {}
autoencoder = {'enc_conv1':[[],[]],'enc_conv2':[[],[]],'enc_conv3':[[],[]],
               'dec_conv1':[[],[]],'dec_conv2':[[],[]],'dec_conv3':[[],[]]}

# Path is the path name ckpt
# Path Example:
path = os.path.dirname(os.path.abspath(__file__)) + '/mobilenet_weight/'
reader = pywrap_tensorflow.NewCheckpointReader(path)

# Var_to_shape_map to store all variable names
var_to_shape_map = reader.get_variable_to_shape_map()
# Can first print all the variable name to see which variables are stored
# print(var_to_shape_map)

for key in var_to_shape_map:
    str_name = key
    
    # Because the model optimization algorithm using Adam, in ckpt generated, containing Tensor, does not need to store related Adam
    if str_name.find('Adam') > -1:
        continue
    if str_name.find('power') > -1:
        continue
    
    if str_name.find('/') > -1:
        names = str_name.split('/')
        layer_name = names[0]
        layer_info = names[1]
    else:
        layer_name = str_name
        layer_info = None
    
    # Kernel and bias layers are convolution kernel parameters and offset parameters, and the type of the specific name of the relevant layer
    if layer_info == 'kernel':
        autoencoder[layer_name][0]=reader.get_tensor(key)
    elif layer_info == 'bias':
        autoencoder[layer_name][1] = reader.get_tensor(key)
    else:
        autoencoder[layer_name] = reader.get_tensor(key)

# save npy
np.save('autoencoder.npy',autoencoder)
print('save npy over...')