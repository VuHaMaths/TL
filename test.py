# Havu: this script is just for checking if the parameters of the new model are the same as from the pretrained
# model

import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

# Havu: print available parameters to check
print(cp.list_variables('save_MertonMDBDP_TL/MertonMDBDPd1nbNeur11nbHL2ndt102eta50BSDE_1'))
print(cp.list_variables('weights/BoundedFNLMDBDPd1nbNeur11nbHL2ndt12030Alpha100BSDE_1'))
print('-------------------------------------------------')

# Havu: print weights of the new model and then of the pretrained model for the first layer, we always take the
# parameters of the last step. Change the step number of the new model, it should always give the same parameters
# for proper transfer learning
# print(cp.load_variable('save/OneAssetMDBDPd2nbNeur12nbHL2ndt82eta50BSDE_1','NetWorkUZNonTrain_2/enc_fc1/weights'))
print(cp.load_variable('save/OneAssetMDBDPd2nbNeur12nbHL2ndt3010eta50BSDE_1','NetWorkUZNonTrain_2/enc_fc1/weights'))
print(cp.load_variable('weights/BoundedFNLMDBDPd2nbNeur12nbHL2ndt12030Alpha100BSDE_1','NetWorkUZ1/enc_fc1/weights'))
#print(cp.load_variable('save/MongeAmpereMDBDPd1nbNeur11nbHL2ndt12030BSDE_1','NetWorkUZ1/enc_fc1/weights'))
print('-------------------------------------------------')

# Havu: print weights of the new model and then of the pretrained model for the second layer
print(cp.load_variable('save_MertonMDBDP_TL/MertonMDBDPd1nbNeur11nbHL2ndt102eta50BSDE_1','NetWorkUZNonTrain_2/enc_fc2/weights'))
print('-------------------------------------------------')
#print(cp.load_variable('save_MertonMDBDP_TL/MertonMDBDPd1nbNeur11nbHL2ndt102eta50BSDE_1','NetWorkUZNonTrain_9/enc_fc2/weights'))
#print('-------------------------------------------------')
#print(cp.load_variable('save/MongeAmpereMDBDPd1nbNeur11nbHL2ndt12030BSDE_1','NetWorkUZ1/enc_fc2/weights'))
#print('-------------------------------------------------')
print(cp.load_variable('weights/BoundedFNLMDBDPd1nbNeur11nbHL2ndt12030Alpha100BSDE_1','NetWorkUZ1/enc_fc2/weights'))