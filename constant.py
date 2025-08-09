#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/22 17:09
@Author  : st
@File    : constant.py
@Project : LLM_and_mlivus_api
@Software: PyCharm
@文件说明: 

"""
import inspect
import os

# 启动进程数量控制
PROCESS_WORKERS = 1

path = inspect.getfile(inspect.currentframe())
abspath = os.path.abspath(path)
# root_path = os.path.dirname(os.path.dirname(abspath))
root_path = os.path.dirname(abspath)
temp_path = os.path.join(root_path, 'data', 'temp')
TEMP_MD_DIR = os.path.join(temp_path, 'md')
if not os.path.exists(TEMP_MD_DIR):
    os.makedirs(TEMP_MD_DIR)
TEMP_PDF_DIR = os.path.join(temp_path, 'pdf')
if not os.path.exists(TEMP_PDF_DIR):
    os.makedirs(TEMP_PDF_DIR)
TEMP_IMAGE_DIR = os.path.join(temp_path, 'ai_pdf_images')
if not os.path.exists(TEMP_IMAGE_DIR):
    os.makedirs(TEMP_IMAGE_DIR)

EMBEDDING_MODEL_LOCAL_DICT = {
    "m3e-base": os.path.join(root_path, "data", 'models', 'embedding', 'm3e-base')
}
# EMBEDDING_MODEL = os.path.join(root_path, "data", 'models', 'embedding', 'm3e-base')

# RERANK_MODEL = os.path.join(root_path, 'data', 'models', 'rerank', 'nlp_rom_passage-ranking_chinese-base')
RERANK_MODEL = '/data/st/code/LLM_and_mlivus_api/data/model/rerank/bge-reranker-v2-m3'

# *******大模型地址*******
# MODEL_DIR = "/home/yinzq/program/qwen_master/model/Qwen-72B"
# MODEL_DIR = "/home/yinzq/program/qwen_master/model/Qwen1.5-72B-Chat-GPTQ-Int4/qwen/Qwen1___5-72B-Chat-GPTQ-Int4/"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-0___5B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-7B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat"
# MODEL_DIR = "/data/st/code/LLM_and_milvus_api/data/model/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# MODEL_DIR = "/data/st/code/LLM_and_milvus_api_copy/data/model/qwen/Qwen1___5-32B-Chat-GPTQ-Int4"
# MODEL_DIR = "/data/st/code/LLM_and_milvus_api/data/model/qwen/Qwen1___5-7B-Chat"
MODEL_DIR = "/data/st/code/LLM_and_milvus_api/data/model/ZhipuAI/glm-4-9b-chat-1m"
# MODEL_DIR = "/data/st/code/LLM_and_milvus_api/data/model/Qwen/Qwen2___5-14B-Instruct"

# *******使用的gpu编号*******
CUDA_VISIBLE_DEVICES = "0,1,2,3"

LOG_LEVEL = "DEBUG"

USE_RERANK = True
USE_TACTICS = True

URL_KEYWORD = "http://localhost:8898/recive_stream_qwen/sig_lj_QA"

# 查询向量数据库url
URL_GET_DATA = "http://localhost:8899/get_milvus_by_ids"
# URL_GET_DATA = "http://172.18.31.8:8899/get_milvus_by_ids"

# 获取pdf文件url-本地上传
#URL_GET_PDF_STREAM_UPLOAD = "http://10.11.10.60:16673/api/v1/ai-search/archive/download"
URL_GET_PDF_STREAM_UPLOAD = "http://10.12.25.3:16673/api/v1/ai-search/archive/download"
# 获取pdf文件url-全文
URL_GET_PDF_STREAM_FULL_TEXT = "https://www.nstl.gov.cn/api/service/nstl/web/execute?target=nstl4.dispatch&function=api/service/dispatch/handle/getFileByArticleInfo"
#URL_GET_PDF_STREAM_FULL_TEXT = "http://10.12.25.3:16673/api/service/dispatch/handle/getFileByArticleInfo"

# *******MongoDB配置参数*******
MONGO_TP = "TEST_16"
# MONGO_TP = "FORMAL"

# # MongDB配置参数
if MONGO_TP == "BJ":  # --------北京测试库-----------
    MONGO_HOST = '10.215.4.35'
    MONGO_PROT = 27017
    MONGO_DATABASE = 'largeModel'
    MONGO_USER = ''
    MONGO_PWD = ''
elif MONGO_TP == "TEST_16":  # 测试 mongodb://admin:istic_nstl_5888@test-mongodb.nstl-dev.com:27017/nstlStorage?authSource=admin
    MONGO_HOST = '10.66.0.16'
    MONGO_PROT = 27017
    MONGO_DATABASE = 'largeModel'
    MONGO_USER = 'root'
    MONGO_PWD = '123456'
elif MONGO_TP == "TEST":  # 测试 mongodb://admin:istic_nstl_5888@test-mongodb.nstl-dev.com:27017/nstlStorage?authSource=admin
    MONGO_HOST = 'test-mongodb.nstl-dev.com'
    MONGO_PROT = 27017
    MONGO_DATABASE = 'largeModel'
    MONGO_USER = 'admin'
    MONGO_PWD = 'istic_nstl_5888'
elif MONGO_TP == "FORMAL":  # 正式  mongodb://admin:istic_nstl_5888@mongodb2.nstl-dev.com:27017,mongodb1.nstl-dev.com:27017,mongodb3.nstl-dev.com:27017/nstlStorage?authSource=admin
    MONGO_HOST = 'mongodb1.nstl-dev.com,mongodb2.nstl-dev.com,mongodb3.nstl-dev.com'
    MONGO_PROT = 27017
    MONGO_DATABASE = 'largeModel'
    MONGO_USER = 'admin'
    MONGO_PWD = 'istic_nstl_5888'

MONGO_COLLECTION = 'PDF_RECOGNITION'
MONGO_COLLECTION_CACHE = "PDF_RECOGNITION_CACHE"

# MONGO_PDF_FILES = 'pdf_files'
# MONGO_PDF_MD_FILES = 'md_files'
# MONGO_PDF_TRANSLATION_FILES = 'pdf_translation_files'

STATUS_RUNNING = 100
STATUS_SUCCESS = 200
STATUS_ERROR = 400
STATUS_TIMEOUT = 500

# 搜索重试次数
SEARCH_RETRY_TIME = 100
SEARCH_SLEEP_TIME = 5

PDF_MOD = "MinerU"
# PDF_MOD = "Pix2Text"

MONGODB_SERVRE_URL = "http://127.0.0.1:8898/mongodb"

URL_COMMON_QA = "http://172.18.31.9:8898/api/other/qa"

# 根据ID查询solr
API_SOLR_ID = "http://10.11.10.60:16673/api/service/search4/hybrid/details"  # 测试：
# API_SOLR_ID = "http://10.12.25.4:16673/api/service/search4/hybrid/details"  #生产：

# 大模型历史对话保存api
API_MESSAGE_SAVE = "http://10.11.10.51:17602/api/v1/ai-search/ai/sessions/assistant/save"
# 大模型历史对话查询api
API_MESSAGE_SEARCH = "http://10.11.10.51:17602/api/v1/ai-search/ai/sessions/multi-round-detail"

LLM_OPENAI_SERVER_DICT = {
    "Qwen/Qwen1.5-14B": {
        "api": "http://172.18.35.17:50018/v1",
        "api_key": "token-82198b77d369bd299fb3834854972313"  # NSTL 单词转换的md5值
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "api": "http://172.18.35.17:50018/v1",
        "api_key": "token-82198b77d369bd299fb3834854972313"  # NSTL 单词转换的md5值
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "api": "http://172.18.31.9:50018/v1",
        "api_key": "token-82198b77d369bd299fb3834854972313"  # NSTL 单词转换的md5值
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "api": "http://10.215.4.18:50018/v1",
        "api_key": "token-abc123"  # NSTL 单词转换的md5值
    }
}

LLM_MODEL_DEFAULT = "Qwen/Qwen2.5-32B-Instruct"

MAX_LEN_HISTORY = 5
