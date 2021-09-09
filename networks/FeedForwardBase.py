# Havu: this class is the DNN, which uses pretrained weights and has freezed layers.
# Check tf_utils.py for more details of the newly created methods
import numpy as np
import tensorflow as tf

from .tf_utils import load_weights_by_names, take_n_layers, remove_vars_from_train_scope


class FeedForwardBase:
    def __init__(
            self, d, layerSize, activation,
            num_layers_to_load_and_freeze: int = None,
            path_saved_checkpoint: str = None,
            weights_step: int = 1):
        self.d = d
        self.layerSize = layerSize
        self.activation = activation

        self._num_layers_to_load_and_freeze = num_layers_to_load_and_freeze
        self._path_saved_checkpoint = path_saved_checkpoint
        self._weights_step = weights_step

    # Havu: method to load pretrained parameters
    def preload_weights(self, iStep, sess: tf.Session, name_of_scope: str = None, weights_path=None) -> bool:
        # Havu: If parameters are in order to load parameters, load them
        if self._num_layers_to_load_and_freeze and \
                self._path_saved_checkpoint is not None:
            # Havu: First - take n variables
            if name_of_scope is not None:
                W_list, b_list = self.getBackWeightAndBias(iStep, name_of_scope=name_of_scope)
            else:
                W_list, b_list = self.getBackWeightAndBias(iStep)
            tf_vars_dict = take_n_layers(W_list, b_list, self._num_layers_to_load_and_freeze, iStep,
                                         weights_step=self._weights_step, name_of_scope=name_of_scope)
            # Havu: Second - Load weights from saved files
            load_weights_by_names(self._path_saved_checkpoint if weights_path is None else weights_path, sess,
                                  tf_vars_dict)
            return True
        return False
    # Havu: method to freeze layers
    def freeze_layers(self, iStep):
        if self._num_layers_to_load_and_freeze and \
                self._path_saved_checkpoint is not None:
            # Havu: First - take n variables
            W_list, b_list = self.getBackWeightAndBias(iStep)
            tf_vars_dict = take_n_layers(W_list, b_list, self._num_layers_to_load_and_freeze, iStep,
                                         weights_step=self._weights_step)
            # Havu: remove variables from trainable collection, which the optimizers (in our case ADAM) uses to
            # determine which parameters to train, the print command is just for controlling purposes (can be excluded
            # for more clarity in the output terminal)
            print('--------------')
            print('Parameters to train before freezing layers: ', tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES))
            remove_vars_from_train_scope(list(tf_vars_dict.values()))
            print('Freeze following parameters: ', list(tf_vars_dict.values()))
            print('================')
            print('Parameters to train after freeze: ', tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES))
            print('--------------')
            return True
        return False
