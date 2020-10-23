from util_html import get_dir_list, html_to_json_zh, html_to_json_it
from config import *

zh_dir_list = get_dir_list(ZH_DIR) # get file path of chinese dataset
it_dir_list = get_dir_list(IT_DIR) # get file path of italian dataset

zh_json = html_to_json_zh(zh_dir_list, ZH_SIZE, ZH_JSON_PATH)
it_json = html_to_json_it(it_dir_list, IT_SIZE, IT_JSON_PATH)
