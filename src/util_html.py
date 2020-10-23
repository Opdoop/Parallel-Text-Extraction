from bs4 import BeautifulSoup
import os
import json
import re
from langdetect import detect
import nltk
from nltk.util import ngrams
import jieba
from config import *
import time


def get_dir_list(in_dir):
    '''
    Return dir list
    :param in_dir: input dir
    :return: dir list
    '''
    files_list = list()
    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        files_list += [os.path.join(dirpath, file) for file in filenames]

    return files_list

# Function to generate n-grams from sentences.
def extract_it_ngrams(data, num):
    '''
    抽取 ngram
    :param data: 字符串
    :param num: n
    :return: n-gram list
    '''
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]

def extract_zh_ngrams(data, num):
    '''
    抽取 ngram
    :param data: 字符串
    :param num: n
    :return: n-gram list
    '''
    seg_list = jieba.cut(data, cut_all=False)
    seg_str = ' '.join(seg_list)
    n_grams = ngrams(nltk.word_tokenize(seg_str), num)
    return [''.join(grams) for grams in n_grams]

def text_check_zh(text):
    '''
    chinese candidates filter
    :param text: input string
    :return: ture or false
    '''
    text_len = len(text)
    if text_len >= 5 and text_len <= 20:
        if not re.search(
                u"[（）()【】“”@《》:：；;0-9a-zA-Z\u30a0-\u30ff\u3040-\u309f\／▽▼＜／——、\[\]\-~#￥%……&*/\|>●「」→★『』～ _＞◎·■　丨◇♀├─＂·〔※◆〜]+",
                text):  # 不包含片假名和平假名，不包含特殊字符英文和数字
            try:
                if (detect(text) == 'zh-cn' or detect(text) == 'zh-tw'):
                    return True
            except:
                pass
                # print(text)
    return False

def text_check_it(text):
    '''
    italian candidates filter
    :param text: input string
    :return: ture or false
    '''
    text_len = len(text)
    if text_len < 100 and text_len > 5:
        if not re.search(u"[\u4e00-\u9fa5、\[\]\-~#￥%……&*/\|“”>●「」→★『』～_＞◎·■　丨◇♀├─＂《》·〔※◆〜0-9:：；;——（）()【】@]+",text):  # 不包含片假名和平假名，不包含特殊字符英文和数字
            try:
                if detect(text) == 'it':
                    return True
            except:
                pass
                # print(text)
    return False


def html_to_json_zh(files_list, bin_size, save_json_dir):
    '''
    读取 html，转为字典 {id: {text:str, file:str, location:str}}
    '''
    ## 获取目录下的所有文件名

    ## 遍历文件，截取字符串
    location_dict = {}
    text_id = 0
    text_set = set()
    file_nbr = 0
    start = time.time()
    for file in files_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:  ## 有些文件可能是反扒， utf8 无法解码
                raw_html = f.read()
        except:
            print(file)  ##  continue，不做任何处理。
            continue
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')  ## new 一个 parser
        except:
            continue
        raw_text = soup.get_text().split('\n')
        for text in filter(None, raw_text):  # 过滤 \n
            text = text.strip()  # 过滤空格
            index = raw_html.find(text)  # 查找下标

            if index != -1:  # 有找到
                ### 对 raw 的长句进行切分，这样感觉比较合理
                for clause in re.split('[,，。.?？!！]', text):
                    if text_check_zh(clause):  # 满足文本条件
                        ## 坐标拼接
                        # pdb.set_trace()
                        try:
                            for place in re.finditer(clause, raw_html):
                                location = '{}:{}'.format(place.start(), place.end())  # 按照提交时候需要的格式存下此句在原 html 中的坐标
                                ## 拼接一条记录   {text:str, file:str, location:str}
                                line_dict = {}
                                line_dict['text'] = clause
                                line_dict['file'] = file[3:]  # 提交格式为 'zh/2020-04-15/balabala.html'， 这里的 file 为项目内结构内的相对路径 '../zh/2020-04-15/balabala.html'，裁剪。
                                line_dict['location'] = location
                                if clause in text_set:
                                    continue
                                ## 并入总表
                                location_dict[text_id] = line_dict
                                text_set.add(clause)
                                text_id += 1  # 更新 id
                                break
                            ## add n-gram
                            # pdb.set_trace()
                            if USE_N_GRAM:
                                for n in [2, 3]:
                                    for n_gram in extract_zh_ngrams(clause, n):
                                        # pdb.set_trace()
                                        if len(n_gram) > 4 and len(n_gram) < 9:
                                            for place in re.finditer(n_gram, raw_html):
                                                location = '{}:{}'.format(place.start(),place.end())  # 按照提交时候需要的格式存下此句在原 html 中的坐标
                                                ## 拼接一条记录   {text:str, file:str, location:str}
                                                line_dict = {}
                                                line_dict['text'] = n_gram
                                                line_dict['file'] = file[3:]  # 提交格式为 'zh/2020-04-15/balabala.html'， 这里的 file 为项目内结构内的相对路径 '../zh/2020-04-15/balabala.html'，裁剪。
                                                line_dict['location'] = location
                                                if n_gram in text_set:
                                                    continue
                                                ## 并入总表
                                                location_dict[text_id] = line_dict
                                                text_set.add(n_gram)
                                                text_id += 1  # 更新 id
                                                break
                        except:
                            continue
                            # print(file)
        if text_id > bin_size:
            break
        file_nbr += 1

    ## 存为 json
    with open(save_json_dir, 'w') as fp:
        json.dump(location_dict, fp)

    end = time.time()
    print('Finish chinese in : {}'.format(end - start))  # log time cost
    return save_json_dir

def html_to_json_it(files_list, bin_size, save_json_dir):
    '''
    读取 html，转为字典 {id: {text:str, file:str, location:str}}
    '''
    ## 获取目录下的所有文件名

    ## 遍历文件，截取字符串
    location_dict = {}
    text_id = 0
    text_set = set()
    file_nbr = 0
    start = time.time()
    for file in files_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:  ## 有些文件可能是反扒， utf8 无法解码
                raw_html = f.read()
        except:
            print(file)  ##  continue，不做任何处理。
            continue
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')  ## new 一个 parser
        except:
            continue
        raw_text = soup.get_text().split('\n')
        for text in filter(None, raw_text):  # 过滤 \n
            text = text.strip()  # 过滤空格
            index = raw_html.find(text)  # 查找下标

            if index != -1:  # 有找到
                ### 对 raw 的长句进行切分，这样感觉比较合理
                for clause in re.split('[,，。.?？!！]', text):
                    # pdb.set_trace()
                    if text_check_it(clause):  # 满足文本条件
                        ## 坐标拼接
                        # pdb.set_trace()
                        try:
                            for place in re.finditer(clause, raw_html):
                                location = '{}:{}'.format(place.start(), place.end())  # 按照提交时候需要的格式存下此句在原 html 中的坐标
                                ## 拼接一条记录   {text:str, file:str, location:str}
                                line_dict = {}
                                line_dict['text'] = clause
                                line_dict['file'] = file[3:]  # 提交格式为 'zh/2020-04-15/balabala.html'， 这里的 file 为项目内结构内的相对路径 '../zh/2020-04-15/balabala.html'，裁剪。
                                line_dict['location'] = location
                                if clause in text_set:
                                    continue
                                ## 并入总表
                                location_dict[text_id] = line_dict
                                text_set.add(clause)
                                text_id += 1  # 更新 id
                                # if text_id % 100 == 0 :
                                #     print(text_id)
                                break
                            ## add n-gram
                            if USE_N_GRAM:
                                for n in [2, 3]:
                                    for n_gram in extract_it_ngrams(clause, n):
                                        if detect(n_gram) == 'it':
                                            for place in re.finditer(n_gram, raw_html):
                                                location = '{}:{}'.format(place.start(),place.end())  # 按照提交时候需要的格式存下此句在原 html 中的坐标
                                                ## 拼接一条记录   {text:str, file:str, location:str}
                                                line_dict = {}
                                                line_dict['text'] = n_gram
                                                line_dict['file'] = file[3:]  # 提交格式为 'zh/2020-04-15/balabala.html'， 这里的 file 为项目内结构内的相对路径 '../zh/2020-04-15/balabala.html'，裁剪。
                                                line_dict['location'] = location
                                                if n_gram in text_set:
                                                    continue
                                                ## 并入总表
                                                location_dict[text_id] = line_dict
                                                text_set.add(n_gram)
                                                text_id += 1  # 更新 id
                                                break
                        except:
                            continue
                            # print(file)

        if text_id > bin_size:
            break
        file_nbr += 1

    ## 存为 json
    with open(save_json_dir, 'w') as fp:
        json.dump(location_dict, fp)

    end = time.time()
    print('Finish italian in : {}'.format(end - start))  # log time cost
    return save_json_dir




