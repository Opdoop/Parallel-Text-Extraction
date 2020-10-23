# Parallel Text Mining
This repo contains the code used by team `肉蛋葱鸡` in the Intermediary of [2020 iFLYTEK text-mining competition](http://challenge.xfyun.cn/topic/info?type=text-mining).

## Results
Results of (chinese, italian) parallel text extraction in Intermediary:

|Rank|Team|Score|
|---|---|---|
|1|肉蛋葱鸡|7562.11126|
|2|====baseline====|6694.19942|
|3|HNwaz8j8x|2394.87172|

## Introduction
Under the restriction of the competition that any additional data or any translation model/API should not be used, we treated this task as an unsupervised pair extraction problem. The crucial part is to project multi-language text into semantic space so that the nearest neighbors could be formed as parallel text pairs.

The main flow has 3 steps. Firstly, extract candidates text from raw html. Secondly, convert text to its vector representation in semantic space. Finally, use `chinese` text as query and search the closest neighbor at `italian` dataset in semantic space. Then use the search results to form (chinese, italian) parallel text pairs. All pairs were sorted by their distance in ascending order.

For text candidates extraction, hand-made rules were used to select candidates from raw html. 

For text vector representation, bert-based model was used to project text to vector. We have tried [xlm-roberta-base](https://huggingface.co/xlm-roberta-base), [m-USE](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3), [LaBSE](https://tfhub.dev/google/LaBSE/1) and so on. LaBSE is the most suitable model in our experiment.

For the nearest neighbor searching, [Scann](https://github.com/google-research/google-research/tree/master/scann) was used to accelerate the searching process as there are millions of chinese and italian text candidates.  

## File Tree
```
/. (root path)
/src (code)
/LaBSE (embedding model)
/mUSE (embedding model)
/text_json (cache result, json for string path in raw html)
/features_hdf5 (cache result, string feature)
/it (it dataset)
/zh (zh dataset)
```

## Environment Setup
The code has been tested on Ubuntu 18.04 using a single GPU. For CPU version and other information, see [Scann](https://github.com/google-research/google-research/tree/master/scann).

### Data
* Get data in [link](http://challenge.xfyun.cn/topic/info?type=text-mining), put it in `it` and `zh` respectively.

### Environment of html parser and embedding model
* Create conda environment for html parser and embedding model
    ```bash
    conda create -n tf2 python=3.6
    conda activate tf2
    ```
* Install html pre-process dependent packages.
    ```bash
    pip install beautifulsoup4 langdetect tqdm nltk jieba numpy==1.18.5
    ```
* Install tensorhub and related packages for sentence embedding model.
    ```bash
    conda install -c anaconda cudatoolkit
    conda install cudnn
  pip install tensorflow_hub bert-for-tf2 tensorflow_text
  ```
  
* Prepare embedding model. Below is example using LaBSE as embedding model. Download sentence embedding model. Then set `LaBSE_PATH` path to `MODEL_PATH` in `./src/config.py`  
    ```bash
    wget -c https://storage.googleapis.com/tfhub-modules/google/LaBSE/1.tar.gz
    tar -xzvf 1.tar.gz -C LaBSE_PATH
    ```

### Environment of scann
* Create new environment for scann. Download scann and install.
    ```bash
    conda deactivate
    conda create -n scann python=3.6
    conda activate scann
    wget https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp36-cp36m-linux_x86_64.whl
    pip install scann-1.0.0-cp36-cp36m-linux_x86_64.whl
    ```


## Quick Start
The main steps were divided into three files for independent testing and adjustment. You could merge all steps into a single file as easy as pie. 

> Note that the default setting was used in the competition. If running the code cost too much time in your machine, you could adjust the experiment setting in `config.py` and searching setting in `utl_scann.py` respectively. 

### 1. Extract Candidates Text from Raw Html
```bash
cd src
conda activate tf2
python run_html_parser.py 
```
This will generate `zh_{EXP_NAME}.json` and `it_{EXP_NAME}.json` in `./text_json`

### 2. Convert Text String to Feature Vectors
```bash
python run_embedding.py
```
This will generate `zh_{EXP_NAME}.hdf5` and `it_{EXP_NAME}.hdf5` in `./features_hdf5`

### 3. Search Matching Candidates and Save results
```bash
conda activate scann
python run_candidates_search.py
```