# !/usr/bin/env python
# -*-coding:utf-8-*-


import os
import random
import numpy
import time
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import scale
from Base import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error,roc_curve
from collections import Counter

def random_projection(data, t):
    result = list()
    rows, cols = data.shape
    for i in range(t):
        random_vector = numpy.array([numpy.random.normal(0, 1, 1)[0] for _ in range(cols)])
        tmp = list()
        for j in range(rows):
            # tmp.append((j, numpy.dot(data[j], random_vector)))
            dot_tmp = 0
            for d_t in range(len(data[j])):
                dot_tmp += data[j][d_t]*random_vector[d_t]
            tmp.append((j, dot_tmp))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        result.append(tmp)
    return result


def fast_voa(data, t):
    rows, cols = data.shape
    n_data = random_projection(data, t)
    f1 = [0]*rows
    for i in range(t):
        cl = [0]*rows
        cr = [0]*rows
        for j in range(rows):
            idx = n_data[i][j][0]
            cl[idx] = j-1
            cr[idx] = rows-j
        for j in range(rows):
            f1[j] += cl[j]*cr[j]
    for i in range(rows):
        f1[i] = (2*numpy.pi*f1[i])/(t*(rows-1)*(rows-2))
    return f1

def result_write(name,label,pred):
    with open('../traditionalresult/fastVOA_result.txt','a') as result:
        result.write('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
                  (name,precision_score(label, pred, average='macro'),
                   recall_score(label, pred, average='macro'),
                   f1_score(label, pred, average='macro'),
                   roc_auc_score(label, pred, ),
                   mean_squared_error(label, pred)
                   ))
        result.write('\n')

if __name__ == '__main__':
    # print "Loading data....."
    #data_sets = ['ECG200', 'Gun_Point', 'HandOutlines', 'Lighting2', 'MoteStrain', 'ECGFiveDays',
    #                 'SonyAIBORobotSurfaceII', 'Strawberry',
    #                 'ToeSegmentation1', 'TwoLeadECG', 'optdigits', 'SyntheticControl']

    dir_path = '../data/UCRtwoclass/'
    for f in os.listdir(dir_path):
        file = dir_path + f
        print(f)
        data = numpy.loadtxt(file, delimiter=',')
        label = data[:, 0]
        data = data[:, 1:]
        rows, cols = data.shape
        print(Counter(label))

        for i in range(len(label)):
            if label[i] != 1:
                label[i] = 0
        data = scale(data)

        hash_num = 100
        start = time.clock()
        score = fast_voa(data, hash_num)
        auc = roc_auc_score(label, score)
        score_ratio,pred = cal_score_ratio(label, score)
        end = time.clock()
        print("Data=%s, AUC=%f, Score_ratio=%f, Time=%f" % (f, auc, score_ratio, (end - start)))

        print('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
              (f, precision_score(label, pred, average='macro'),
               recall_score(label, pred, average='macro'),
               f1_score(label, pred, average='macro'),
               roc_auc_score(label, pred, ),
               mean_squared_error(label, pred)
               ))
        # result_write(f, label, pred)
        # res.write("Data Set=%s, AUC=%f, Time=%f\n" %(set, auc, (end-begin)))




