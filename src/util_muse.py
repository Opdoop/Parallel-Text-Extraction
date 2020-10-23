import tensorflow_text
import tensorflow_hub as hub
import json
import numpy as np
from tqdm import tqdm
import h5py
from config import *

embedding = MODEL_PATH

def json_to_hdf5(json_path, dataset_name, h5f_path):
    embed = hub.load(embedding)

    ## 读取数据
    with open(json_path) as f:
        data = json.load(f)

    ## 创建数组
    dim_row = len(data)  # 行数等于样本个数
    dim_col = 512  # 语言模型的数据向量长度
    matrix = np.zeros((dim_row, dim_col), dtype='float32')  # 创建矩阵

    ## 填入特征值
    for key, item in tqdm(data.items()):
        if int(key) == dim_row:
            break
        text = item['text']
        hidden_state = embed(text)
        # pdb.set_trace()
        matrix[int(key)] = hidden_state

    ## 存为 hdf5 文件
    h5f = h5py.File(h5f_path, "w")
    h5f.create_dataset(dataset_name, data=matrix)
    h5f.close()
    return h5f_path