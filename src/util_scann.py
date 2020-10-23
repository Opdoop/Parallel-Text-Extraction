import numpy as np
import h5py
import time
import scann
import csv
import json
import pdb

def write_to_csv(zh_json, zh_hdf5, ja_json, ja_hdf5, csv_file_name):
    ## 中文，作为 queary
    with open(zh_json) as f:
        zdata = json.load(f)
    query_h5f = h5py.File(zh_hdf5, 'r')
    queries = query_h5f['zh']

    ## 日文，作为 dataset
    with open(ja_json) as f:
        jdata = json.load(f)
    dataset_h5f = h5py.File(ja_hdf5, 'r')
    dataset = dataset_h5f['it']

    # pdb.set_trace()
    ## 创建 Searcher
    normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

    searcher = scann.ScannBuilder(normalized_dataset, 10, "dot_product").tree(
    num_leaves=5000, num_leaves_to_search=200, training_sample_size=750000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).create_pybind()

    ## brute_force
    # searcher = scann.ScannBuilder(normalized_dataset, 10, "dot_product").tree(  # dot_product /  squared_l2
    # num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_brute_force(True).create_pybind()


    print("Start to search...")
    start = time.time()
    neighbors, distances = searcher.search_batched(queries, final_num_neighbors=1)
    end = time.time()

    print("Search Time:", end - start)

    ## 给下标排序，得到 source/target 对应下标
    dis_top = distances[:,:5] # top-1 的 distance
    # pdb.set_trace()
    indexs = np.argsort(dis_top, axis=0) # 按 top-1 的 distance 排序
    save_indexs = indexs[::-1]

    # pdb.set_trace()
    save_neighbors = neighbors[save_indexs, :] # 获取所有 top-20 的 neighbor id
    source_index = save_indexs
    target_index = save_neighbors
    nbr_row, _ = source_index.shape

    zh_indices = list()
    ja_indices = list()
    start = time.time()
    for i in range(nbr_row):
        # pdb.set_trace()
        _source_index = source_index[i][0]
        _target_index = target_index[i][0]

        zh_indices.append(_source_index)
        ja_indices.append(_target_index)

    ## 保存结果
    submit_file = open(csv_file_name, 'w', newline='', encoding='utf8')
    writer = csv.writer(submit_file, delimiter=',')
    writer.writerow(['file_source', 'location_source', 'file_target', 'location_target']) # headline

    ## 遍历获取匹配结果
    nbr_row = min(30000, len(zh_indices))
    for row in range(nbr_row):
        zh_key = str(zh_indices[row])
        file_source = zdata[zh_key]['file']
        location_source = zdata[zh_key]['location'] # 'file', 'location', 'text'

        ja_key = str(ja_indices[row][0])
        file_target = jdata[ja_key]['file']
        location_target = jdata[ja_key]['location']

        writer.writerow([file_source, location_source, file_target, location_target])

    ## 关闭文件
    submit_file.close()
