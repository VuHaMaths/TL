import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp


print(cp.list_variables('save/MertonPWGd1nbNeur11nbHL2ndt30eta50BSDE_1'))
print('-------------------------------------------------')
print(cp.load_variable('save/MertonPWGd1nbNeur11nbHL2ndt30eta50BSDE_1','NetWorkUZ1/enc_fc1/weights'))
print(cp.load_variable('saved_parameters/FNL_PWG_1/BoundedFNLPWGd1nbNeur11nbHL2ndt120Alpha100BSDE_1','NetWorkUZ1/enc_fc1/weights'))