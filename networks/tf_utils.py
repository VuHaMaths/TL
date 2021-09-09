import traceback

import tensorflow as tf
from typing import Dict, Tuple, List


# Havu:
# tf_weights, tf_bises - can be called via the method `getBackWeightAndBias`
# Which return weights and biases as list - these variables are inputs
# of the method `take_n_layers`
#
# Number of biases and weights must be the same
#
# Algorithm: create Neural Network (take weights and biases) -> put into `take_n_layers` ->
# put into `load_weights_by_names`, load certain weights via checkpoint ->
# remove params from train scope, i.e. call `remove_vars_from_train_scope`
#

# Havu: method to determine how many layers to freeze
def take_n_layers(tf_weights: list, tf_bises: list, n: int, iStep: int, weights_step: int, name_of_scope: str = None) -> \
Dict[str, tf.Variable]:
    assert len(tf_weights) >= n and len(tf_bises) >= n
    take_n_weights, take_n_biases = tf_weights[:n], tf_bises[:n]
    n_vars_list = take_n_weights + take_n_biases
    tf_vars_dict = dict([(s_var.name.split(':')[0], s_var) for s_var in n_vars_list])
    new_dict = dict()
    for key, val in tf_vars_dict.items():
        net_name = key.split('/')[0].split(str(iStep))[0]
        varname = "/".join(key.split('/')[1:])
        varname = net_name + str(weights_step) + '/' + varname
        new_dict[varname] = val
    return new_dict


# Havu: method to load parameters
def load_weights_by_names(path_weights: str, sess: tf.Session, var_dict: dict):
    try:
        saver = tf.train.Saver(var_dict)
        saver.restore(sess, path_weights)
        print('Parameters are loaded.')
    except Exception as ex:
        print('path_weights', path_weights)
        print('var_dict', var_dict)
        reader = tf.train.NewCheckpointReader(path_weights)
        info = reader.get_variable_to_shape_map()
        print('vars in checkpoint')
        for var in info.keys():
            print(var)
        traceback.print_exc()
        raise ex


# Havu: method to remove trainable parameters from the trainable collection, which the optimizer(in our case ADAM)
# uses to determine which parameters to train
def remove_vars_from_train_scope(tf_variables_list: list):
    train_vars_list = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    for single_tf_variable in tf_variables_list:
        train_vars_list.remove(single_tf_variable)

    print('Parameters are removed from training!')

# Havu: error message, if saved file does not exists
def load_tensor_from_ckpt(ckpt_path, tensor_name):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    try:
        tensor = reader.get_tensor(tensor_name)
        return tensor
    except Exception as ex:
        print('Error. ')
        print(ex)
        traceback.print_exc()
        print(f'Did not found tensor with name={tensor_name} in checkpoint with'
              f' path={ckpt_path}.')
        info = reader.get_variable_to_shape_map()
        print(f'Available tensor names are:')
        for tensor_name in info.keys():
            print(tensor_name)
        raise Exception('Failed loading tensor from checkpoint. For details please see'
                        ' the info above.')
