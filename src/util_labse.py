import json
import bert
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from config import MODEL_PATH

MODEL_URL =  MODEL_PATH # set the local catch path of model
max_seq_length = 64  # set max lenght


def get_model(model_url, max_seq_length):
    '''
    reference: https://tfhub.dev/google/LaBSE/1
    :param model_url:
    :param max_seq_length:
    :return:
    '''
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    # LaBSE layer.
    pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

    # Define model.
    return tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=pooled_output), labse_layer


# reference: https://tfhub.dev/google/LaBSE/1
labse_model, labse_layer = get_model(
    model_url=MODEL_URL, max_seq_length=max_seq_length)
# reference: https://tfhub.dev/google/LaBSE/1
vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


def create_input(input_strings, tokenizer, max_seq_length):
    '''
    reference: https://tfhub.dev/google/LaBSE/1
    :param input_strings:
    :param tokenizer:
    :param max_seq_length:
    :return:
    '''
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
        # Tokenize input.
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

    # Padding or truncation.
    if len(input_ids) >= max_seq_length:
        input_ids = input_ids[:max_seq_length]
    else:
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

    input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

    input_ids_all.append(input_ids)
    input_mask_all.append(input_mask)
    segment_ids_all.append([0] * max_seq_length)

    return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def encode(input_text):
    '''
    reference: https://tfhub.dev/google/LaBSE/1
    :param input_text:
    :return:
    '''
    input_ids, input_mask, segment_ids = create_input(
        input_text, tokenizer, max_seq_length)
    return labse_model([input_ids, input_mask, segment_ids])


def json_to_hdf5(json_path, dataset_name, h5f_path):
    ## 读取数据
    with open(json_path) as f:
        data = json.load(f)

    ## 创建数组
    dim_row = len(data)  # 行数等于样本个数
    dim_col = 768  # 语言模型的数据向量长度
    matrix = np.zeros((dim_row, dim_col), dtype='float32')  # 创建矩阵

    ## 填入特征值
    for key, item in tqdm(data.items()):
        if int(key) == dim_row:
            break
        text = item['text']
        hidden_state = encode([text])
        # pdb.set_trace()
        matrix[int(key)] = hidden_state

    ## 存为 hdf5 文件
    h5f = h5py.File(h5f_path, "w")
    h5f.create_dataset(dataset_name, data=matrix)
    h5f.close()
    return h5f_path
