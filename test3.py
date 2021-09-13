import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

print(cp.load_variable('saved_parameters/FNL_MDBDP_2/BoundedFNLMDBDPd2nbNeur12nbHL2ndt12030Alpha100BSDE_1','NetWorkUZ1/enc_fc1/weights'))
print(cp.load_variable('save/OneAssetMDBDPd2nbNeur12nbHL2ndt3010eta50BSDE_1','NetWorkUZ1/enc_fc1/weights'))