import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

print(cp.list_variables('save_Gnet/MertonPWGd1nbNeur11nbHL2ndt10eta50BSDE_0'))
print(cp.list_variables('save_Gnet/MertonPWGd1nbNeur11nbHL2ndt10eta50BSDE_9'))
print(cp.load_variable('save_Gnet/MertonPWGd1nbNeur11nbHL2ndt10eta50BSDE_1','NetWorkUZ1/enc_fc1/weights'))
print(cp.load_variable('save_Gnet/MertonPWGd1nbNeur11nbHL2ndt10eta50BSDE_9','NetWorkUZ9/enc_fc1/weights'))