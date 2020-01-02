# !/usr/bin/env python
# -*-coding:utf-8-*-
import numpy as np
from collections import Counter

def read_config(name):
    obj = open(name, 'r')
    config = dict()
    for line in obj.readlines():
        tmp = line[:-1].split(':')
        data = tmp[0].split('=')[1]
        k = tmp[1].split('=')[1]
        config[data] = int(k)
    obj.close()
    return config

# anomaly score ratio E(PAPR)
# def cal_score_ratio(label, predict):
#     score_n, score_a = 0, 0
#     n_count, a_count = 0, 0
#     print(label)
#     print(predict)
#     exit()
#     for p in range(len(label)):
#         if label[p] == 1:
#             score_n += predict[p]
#             n_count += 1
#         else:
#             score_a += predict[p]
#             a_count += 1
#     score_m = (score_n+score_a)/(n_count+a_count)
#     score_a_m = score_a/a_count
#     score_ratio = score_a_m/score_m
#     return score_ratio

def cal_score_ratio(label, predict):
    c = Counter(label)
    true_ano = c[0] / (c[1] + c[0])
    # print(true_ano)
    anomaly_score = np.percentile(predict, (1 - true_ano) * 100)
    # print(anomaly_score)
    result = map(lambda x: 0 if x < anomaly_score else 1, predict)
    result = list(result)
    # print(result)


    # anomaly_num = 0
    # for l in label:
    #     if l == 0:
    #         anomaly_num += 1
    # print(anomaly_num)
    # print(len(label))
    #
    #
    # arg_anomaly_index = np.argsort(predict) #将x中的元素从小到大排列，提取其对应的index(索引号)
    # top_anomaly_index = arg_anomaly_index[:anomaly_num]
    # real_anomaly_index = list()
    # for t_a_i in top_anomaly_index:
    #     if label[t_a_i] != 1:
    #         real_anomaly_index.append(t_a_i)
    # real_anomaly_score = np.mean(predict[real_anomaly_index])
    # all_anomaly_score = np.mean(predict)
    # print(real_anomaly_score)
    # print(all_anomaly_score)
    # exit()
    return anomaly_score,result