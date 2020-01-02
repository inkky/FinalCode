# !/usr/bin/env python
# -*-coding:utf-8-*-


import os
import numpy
import time
from Base import *
from sklearn.metrics import roc_auc_score


def preprocessing(data, k):
    # 每个序列分成k段
    new_data = list()
    rows, cols = data.shape
    width = cols/k
    for r in range(rows):
        line = list()
        for i in range(k):
            data_tmp = data[r, i*width:(i+1)*width]
            x_min, x_max = min(data_tmp), max(data_tmp)
            boundary = 0.2*(x_max-x_min)
            p_min, p_max = 0, 0
            for d_t in data_tmp:
                if d_t < x_min+boundary:
                    p_min += 1
                elif d_t > x_max-boundary:
                    p_max += 1
            line.append((x_min, x_max, 1.0*p_min/width, 1.0*p_max/width))
        new_data.append(line)
    return new_data


def cal_score(n_data, k):
    d_len = len(n_data)
    dist = list()
    for r in range(d_len):
        dist_tmp = 0
        for c in range(d_len):
            if r == c:
                continue
            for d in range(k):
                dist_tmp += dist_interval(n_data[r][d], n_data[c][d])
        dist_tmp /= d_len
        dist.append(dist_tmp)
    scores = list()
    for r in range(d_len):
        score_tmp = 0
        for c in range(d_len):
            score_tmp += (dist[r]-dist[c])**2
        scores.append(score_tmp)
    return scores


def dist_interval(val1, val2):
    dist = 0
    x_min, x_max = val1[0], val1[1]
    y_min, y_max = val2[0], val2[1]
    # dist += float(y_min-x_max)/(y_max-x_min+0.001)
    if max(x_max, y_max) != min(x_min, y_min):
        dist += float(min(x_max, y_max)-max(x_min, y_min))/(max(x_max, y_max)-min(x_min, y_min))
    dist += numpy.exp(-((val1[2]-val2[2])**2+(val1[3]-val2[3])**2)**0.5)
    return dist/2


if __name__ == '__main__':
    # data_sets = ['ECG200', 'Gun_Point', 'HandOutlines', 'Lighting2', 'MoteStrain', 'ECGFiveDays', 'SonyAIBORobotSurfaceII', 'Strawberry',
    #              'ToeSegmentation1', 'TwoLeadECG', 'optdigits', 'SyntheticControl']

    dir_path = '../data/'
    conf = read_config('../config/Interval')

    for set in os.listdir(dir_path):
        if set != 'DiatomSizeReduction_3_1':
            continue

        file = dir_path + set
        data = numpy.loadtxt(file, delimiter=',')
        labels = data[:, 0]
        data = data[:, 1:]
        d_rows, d_cols = data.shape
        for i in range(d_rows):
            if labels[i] != 1:
                labels[i] = 0
        k = conf.get(set)
        try:
            begin = time.clock()
            n_data = preprocessing(data, k)
            scores = cal_score(n_data, k)
            end = time.clock()
            auc = 1-roc_auc_score(labels, scores)
            output = 'Data=%s, AUC=%f, Time=%f' %(set, auc, (end-begin))
            print(output)
        except:
            print(set, k, ' Error')
        # break

