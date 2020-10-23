ZH_DIR = '../zh'
IT_DIR = '../it'

EXP_NAME = 'clause-sent' # run name, for save cache and resutl file

ZH_JSON_PATH = '../text_json/zh_{}.json'.format(EXP_NAME) # candidates path
ZH_SIZE = 1200000 # maximum candidate sentence number of zh
IT_JSON_PATH = '../text_json/it_{}.json'.format(EXP_NAME) # candidates path
IT_SIZE = 3000000 # maximum candidate sentence number of it

USE_N_GRAM = True # if extract string n-gram

ZH_HDF5_PATH = '../features_hdf5/zh_{}.hdf5'.format(EXP_NAME) # candidates feature path
IT_HDF5_PATH = '../features_hdf5/it_{}.hdf5'.format(EXP_NAME) # candidates feature path

MODEL = 'LaBSE'  # LaBSE / mUSE
MODEL_PATH = '/home/ifly/Parallel-Text-Extraction/LaBSE' # "'/home/ifly/Parallel-Text-Extraction/mUSE-Large"

RESULT_PATH = '{}.csv'.format(EXP_NAME) # csv results path