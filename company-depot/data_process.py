import csv
from tqdm import tqdm as tqdm
import features as fea
import random
import numpy as np
import pickle
from config import *
import Levenshtein
import heapq
from multiprocessing.pool import Pool

import pymysql

import elasticsearch
import json

es = elasticsearch.Elasticsearch(es_host,timeout = 3000)
db = pymysql.connect(
            host="10.10.10.10",
            user="readonly",password="readonly",
            database='mag-190302',
            charset='utf8mb4')
cursor = db.cursor()
sql = "SELECT PaperCount, NormalizedName FROM `mag-190302`.Affiliations;"
cursor.execute(sql)
ret = cursor.fetchall()

nor2popular = {}
for i in range(len(ret)):
    nor2popular[ret[i][1]] = ret[i][0]
    
    

def get_Top_K_doc(es, index,words, K=10):
#     search_words = words.split()
    ret = es.search(index=index, body={
          "size": K,
          "sort": [
            {
              "_score": {
                "order": "desc"
              }
            }
          ],
          "_source": {
            "excludes": []
          },
          "stored_fields": [
            "*"
          ],
          "script_fields": {},
          "docvalue_fields": [],
          "query": {
    "match": {
      "original_name": words
    }
  }
        })
    if len(ret['hits']['hits']) <= 0:
        return []
    else:
        search_result_list_score = []
        search_result_list_norm = []
        search_result_list_ori = []
        for i, k in enumerate(ret['hits']['hits']):
            #print(i, k['_source']['normalized_name'])
            search_result_list_norm.append(k['_source']['normalized_name'])
            search_result_list_ori.append(k['_source']['original_name'])
            search_result_list_score.append(k['_score'])
        return search_result_list_norm[:K], search_result_list_ori[:K], search_result_list_score[:K]
        
l_test = "/home/datamerge/SIGIR/Data/processed_mag_50_500_test_data/mag_l_50_500_test_data.txt"
h_test = "/home/datamerge/SIGIR/Data/processed_mag_50_500_test_data/mag_h_50_500_test_data.txt"
m_test = "/home/datamerge/SIGIR/Data/processed_mag_50_500_test_data/mag_m_50_500_test_data.txt"
test_file = "/home/datamerge/SIGIR/Data/Test/new_mag_test.txt"

#train_data
train_data = []
file = test_file
with open(file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        words = line.strip().split('\t\t')
        if len(words) == 3:
            train_data.append([words[0],words[1],words[2]])
            
            
def get_result(input_data):
    train_data = input_data[0]
    i = input_data[1]
    index_es = 'new_cd1_mag_doc'
    features_list = []
    query = train_data[0]
    nor_result = train_data[1]
    try:
        infor = get_Top_K_doc(es,index_es,query,100)
    except:
        return []
    if infor == []:
        return []
    nor = infor[0]
    ori = infor[1]
    score = infor[2]
    candicate = []
    if len(ori) < 10:
        for index in range(len(ori)):
            candicate.append([nor[index],score[index],ori[index]])
    else:
        for index in range(10):
            candicate.append([nor[index],score[index],ori[index]])

    distances = [Levenshtein.distance(query,index) for index in ori]
    if len(distances) == 1:
        candicate.append([nor[0],score[0],ori[0]])
    else:
        max_num = list(map(distances.index, heapq.nlargest(2, distances)))
        candicate.append([nor[max_num[0]],score[max_num[0]],ori[max_num[0]]])
        candicate.append([nor[max_num[1]],score[max_num[1]],ori[max_num[1]]])

    distances = [Levenshtein.distance(fea.calibration(query).replace(' ',''),fea.calibration(index).replace(' ','')) for index in ori]
    if len(distances) == 1:
        candicate.append([nor[0],score[0],ori[0]])
    else:
        max_num = list(map(distances.index, heapq.nlargest(2, distances)))
        candicate.append([nor[max_num[0]],score[max_num[0]],ori[max_num[0]]])
        candicate.append([nor[max_num[1]],score[max_num[1]],ori[max_num[1]]])

    for index in range(len(candicate)):
#        features
        temp = {}
        if candicate[index][0] == nor_result:
            temp['label'] = 5
        else:
            temp['label'] = 0
        temp['qid'] = i
        feature = []
        feature.extend(fea.get_length(query))
        feature.extend(fea.extract_country(query))
        feature.extend([candicate[index][1]])
        feature.extend(fea.query_nor_equal(query,candicate[index][0]))
        feature.extend(fea.query_ori_equal(query,candicate[index][2]))
        feature.extend(fea.prefix_suffix(query,candicate[index][0]))
        feature.extend(fea.common_words(query,candicate[index][0]))
        feature.extend(fea.get_distance(query,candicate[index][0],candicate[index][2]))
        feature.extend(fea.get_Jaccard(query,candicate[index][0],candicate[index][2]))
        feature.extend(fea.country_same(query,candicate[index][0]))
        if candicate[index][0] in nor2popular.keys():
            feature.extend([nor2popular[candicate[index][0]]])
        else:
            feature.extend([0])
        feature.extend([len(candicate[index][0])])
        feature.extend(fea.extract_country(candicate[index][0]))
        temp['features'] = feature
        features_list.append(temp)
    return features_list, candicate
            
with Pool() as pool:
    features_result = list(tqdm(pool.imap(get_result,[[train_data[i],i] for i in range(len(train_data))] ),total = len(train_data)))
    
def write_data(data,output_file):
    with open(output_file,'w') as f:
        for d in data:
            f.write(str(d['label']))
            f.write(' ' + 'qid:')
            f.write(str(d['qid']))
            f.write(' ')
            for i in range(len(d['features'])):
                f.write(str(i+1))
                f.write(':')
                f.write(str(d['features'][i]))
                f.write(' ')
            f.write('\n')
            
write_data(features_list,'./result/svm.txt')