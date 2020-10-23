from config import *
if MODEL == 'LaBSE':
    from util_labse import json_to_hdf5
if MODEL == 'mUSE':
    from util_muse import json_to_hdf5

zh_hdf5 = json_to_hdf5(ZH_JSON_PATH, 'zh', ZH_HDF5_PATH)
it_hdf5 = json_to_hdf5(IT_JSON_PATH, 'it', IT_HDF5_PATH)

