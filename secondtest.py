import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp


print(cp.list_variables('save/MertonMDBDPd1nbNeur11nbHL2ndt102eta50BSDE_1'))
print('-------------------------------------------------')
print(cp.load_variable('save/OneAssetMDBDPd2nbNeur12nbHL2ndt22eta50BSDE0','NetWorkGamNotTrain_1/enc_fc1/weights'))
print(cp.load_variable('save_bounded_fnlmdbdp/BoundedFNLMDBDPd2nbNeur12nbHL2ndt22Alpha100BSDE_1','NetWorkGamNotTrain_1/enc_fc1/weights'))
print('-------------------------------------------------')

print(cp.load_variable('save/OneAssetMDBDPd2nbNeur12nbHL2ndt22eta50BSDE0','NetWorkGamNotTrain_1/enc_fc2/weights'))
print(cp.load_variable('save_bounded_fnlmdbdp/BoundedFNLMDBDPd2nbNeur12nbHL2ndt22Alpha100BSDE_1','NetWorkGamNotTrain_1/enc_fc2/weights'))
print('-------------------------------------------------')