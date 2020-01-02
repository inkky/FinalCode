# !/usr/bin/env python
# -*-coding:utf-8-*-

import os
import time
import numpy
import warnings
from sklearn.metrics import roc_auc_score,mean_squared_error,recall_score
from sklearn.preprocessing import scale,minmax_scale
from Base import *

# warnings.filterwarnings('error')

splits = dict()
splits[2] = [0, float('inf')]
splits[3] = [-0.43, 0.43, float('inf')]
splits[4] = [-0.67, 0, 0.67, float('inf')]
splits[5] = [-0.84, -0.25, 0.25, 0.84, float('inf')]
splits[6] = [-0.97, -0.43, 0, 0.43, 0.97, float('inf')]
splits[7] = [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, float('inf')]
splits[8] = [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, float('inf')]
splits[9] = [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, float('inf')]
splits[10] = [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28, float('inf')]


def preprocessing(data, k):
    split = splits[k]
    split_index = range(k)
    rows, cols = data.shape
    new_data = numpy.zeros((k, cols))
    for i in range(rows):
        seg = data[i]
        for j in range(cols):
            for s in split_index:
                if seg[j] < split[s]:
                    new_data[s, j] += 1
                    break
    return new_data


def get_seg_info(k):
    center = numpy.zeros(k)
    width = numpy.zeros(k)
    segs = splits[k][:]
    segs.insert(0, -3.0)
    segs[-1] = 3.0
    while k > 0:
        center[k-1] = (segs[k]+segs[k-1])/2
        # width[k-1] = (segs[k]-segs[k-1])/2
        width[k-1] = abs(segs[k] - center[k-1])
        k -= 1
    return center, width


def neigh_score(point, time, s, k, center, width, table):
    if s == 0:
        return width[1]/abs(point-center[1]) * table[1][time]
    elif s == k-1:
        return width[k-2]/abs(point-center[k-2]) * table[k-2][time]
    else:
        pre = width[s-1]/abs(point-center[s-1]) * table[s-1][time]
        after = width[s+1]/abs(point-center[s+1]) * table[s+1][time]
        return (pre+after)/2


def neigh_score2(point, time, s, k, center, width, table):
    if point < center[s]:
        if s > 0:
            return width[s-1]/abs(point-center[s-1]) * cal_val(table[s-1][time])
        else:
            return 0
    else:
        if s != k-1:
            return width[s+1]/abs(point-center[s+1]) * cal_val(table[s+1][time])
        else:
            return 0


def cal_val(x):
    return x


def cal_score(test_data, table, cols, k):
    center, width = get_seg_info(k)
    predict = list()
    for item in test_data:
        score = 0
        for p in range(cols):
            s = 0
            while s < k:
                if item[p] < splits[k][s]:
                    tmp = cal_val(table[s][p]) + neigh_score(item[p], p, s, k, center, width, table)
                    score += tmp
                    break
                s += 1
        score = 1.0 * score
        predict.append(score)
    return numpy.array(predict)

from collections import Counter
if __name__ == '__main__':

    dir_path = '../data/UCRtwoclass/'
    config = read_config('config/EITable')
    # print config
    # exit()

    for f in os.listdir(dir_path):
        f_path = dir_path + f
        # if os.path.isdir(f_path):
        #     continue
        # if f != 'ECG200':
        #     continue

        test_data = numpy.loadtxt(f_path, delimiter=',')
        label = test_data[:, 0]
        test_data = test_data[:, 1:]
        rows, cols = test_data.shape
        label_count = dict()
        for i in range(rows):
            label_count[label[i]] = label_count.get(label[i], 0)+1
        test_data = scale(test_data, axis=1)

        # k = config.get(f)
        k=8
        center, width = get_seg_info(k)
        begin = time.clock()
        t_data = preprocessing(test_data, k)
        predict= minmax_scale(cal_score(test_data, t_data, cols, k))
        end = time.clock()

        # auc = roc_auc_score(label, predict)
        score_ratio, pred = cal_score_ratio(label, predict)
        error = mean_squared_error(label, pred)
        auc=recall_score(label, pred, average='macro')
        # score_ratio = cal_score_ratio(label, predict)
        print("Data=%s, AUC=%f, error=%f, Score_ratio=%f, Time=%f" % (f, auc, error,score_ratio, (end - begin)))

