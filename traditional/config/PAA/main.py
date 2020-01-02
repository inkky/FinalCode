# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 14:26
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
5.9 update
computer.txt中只用bt的准确率高于所有，很奇怪
'''
import os
import sys
import numpy
import operator
import random
import numpy as np
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from traditional.PAA.expand_PAA import *
from traditional.SAX.sax_variant import *
from collections import Counter

def data_scale(data):
    """归一化"""
    # print('data_scale')
    standscaler = StandardScaler()
    mscaler = MinMaxScaler(feature_range=(0, 1))
    data = standscaler.fit_transform(data)
    data = mscaler.fit_transform(data)
    return data

def result_write(name,best_win,precision,recall,f1,auc,error,t):
    with open('PAAresult/TPAA_result_2.txt','a') as result:
        if auc<0.5:
            auc=1-auc
        # result.write('name: %s, win_size: %d, Precision: %f, Recall: %f, F1: %f, auc:  %f, error:  %f , time:  %f' % \
        #       (name, best_win, precision, recall, f1, auc, error, t))
        result.write('%s, %f' % \
                           (name, auc))

        result.write('\n')


def main():

    # file=['BeetleFly.txt']
    # file = ['Computers.txt', 'Earthquakes.txt']
    file = ['Earthquakes.txt']
    # file = ['Gun_Point.txt', 'Ham.txt']
    # file = ['BeetleFly.txt','Coffee.txt','ECG200.txt','Herring.txt', 'Lighting2.txt']
    # file = ['MoteStrain.txt', 'Strawberry.txt']
    # file = ['ToeSegmentation2.txt', 'Wine.txt']


    filedir = '../../data/UCRtwoclass/'
    # for name in sorted(os.listdir(filedir)):
    for name in file:
        print(name)
        data = np.loadtxt('../../data/UCRtwoclass/'+name, delimiter=',')
        labels = data[:, 0]
        c = Counter(labels)
        print(c)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))
        labels = np.where(labels == c[1][1], 1, 0)
        data = data[:, 1:]
        rows, cols = data.shape
        k=3

        max_auc=0
        # win_size=3
        alphabetSize=3
        for win_size in range(3,20):
            trainBeginTime = time.clock()

            # raw_dist, pred = BT_PAA(win_size, data, labels, k)
            raw_dist, pred = TPAA(win_size, data, labels, k)
            # raw_dist, pred = Raw_PAA(win_size, data, labels, k)
            # raw_dist, pred = Cosine_PAA(win_size, data, labels, k)
            # raw_dist, pred = EU(win_size, data, labels, k)
            # raw_dist, pred = esax(data, win_size, alphabetSize, k, labels)

            trainEndTime = time.clock()
            for i in range(rows):
                if labels[i] != 1:
                    labels[i] = 0

                if pred[i] != 1.0:
                    pred[i] = 0
            last_time=trainEndTime - trainBeginTime
            auc=roc_auc_score(labels, pred)

            if max_auc<auc:
                max_auc=auc
                best_win=win_size
                precision=precision_score(labels, pred, average='macro')
                recall=recall_score(labels, pred, average='macro')
                f1=f1_score(labels, pred, average='macro')
                t=last_time
                error=mean_squared_error(labels, pred)
        print('name: %s, win_size: %d, Precision: %f, Recall: %f, F1: %f, auc:  %f, error:  %f , time:  %f' % \
              (name,best_win,precision,recall,f1,max(auc,(1-auc)),error,t))
        result_write(name,best_win,precision,recall,f1,auc,error,t)



if __name__ == '__main__':
    print('bt_paa')
    main()
