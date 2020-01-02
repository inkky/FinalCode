#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lenovo on 2019/5/13


import time
import matplotlib
from traditional.SAX.sax_variant import notify_result,esax,original_sax,sax_sd,sax_td,tsax
from traditional.SAX.sax_knn import sax
import os
from collections import Counter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error


def result_write(name,best_win,precision,recall,f1,auc,error,t):
    with open('../SAXresult/ESAX_result.txt','a') as result:
        result.write('name: %s, win_size: %d, Precision: %f, Recall: %f, F1: %f, auc:  %f, error:  %f , time:  %f' % \
              (name, best_win, precision, recall, f1, auc, error, t))
        result.write('\n')

def main():
    filedir='../../data/UCRtwoclass/'
    for name in os.listdir(filedir):
        print(name)
        data = np.loadtxt('../../data/UCRtwoclass/'+name, delimiter=',')
        labels = data[:, 0]
        c=Counter(labels)
        print(c)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))
        labels = np.where(labels == c[1][1], 1, 0)
        data = data[:, 1:]
        rows, cols = data.shape

        # parameters
        k=3
        alphabetSize = 3
        max_auc = 0

        #win_size是段中时间点的个数
        for win_size in range(4,20):
            trainBeginTime = time.clock()
            raw_dist, pred = esax(data, win_size, alphabetSize, k, labels)
            trainEndTime = time.clock()
            preanomaly = list()
            trueanomaly = list()
            # for i in range(rows): #1为normal ，0 为abnormal
            #     if labels[i] != 1.0:
            #         labels[i] = 0
            #         trueanomaly.append(i)

                # if pred[i] != 1.0:
                #     pred[i] = 0
                #     preanomaly.append(i)

            # result

            last_time = trainEndTime - trainBeginTime
            auc = roc_auc_score(labels, pred)

            if max_auc < auc:
                max_auc = auc
                best_win = win_size
                precision = precision_score(labels, pred, average='macro')
                recall = recall_score(labels, pred, average='macro')
                f1 = f1_score(labels, pred, average='macro')
                t = last_time
                error = mean_squared_error(labels, pred)
        print('name: %s, win_size: %d, Precision: %f, Recall: %f, F1: %f, auc:  %f, error:  %f , time:  %f' % \
              (name, best_win, precision, recall, f1, auc, error, t))
        result_write(name, best_win, precision, recall, f1, auc, error, t)


if __name__ == '__main__':
    # print('esax')
    main()