# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 16:32
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''

'''
import numpy as np
import pandas as pd
import sax_knn
import operator
import time

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,label_ranking_average_precision_score,label_ranking_loss,coverage_error

import matplotlib.pyplot as plt

np.random.seed(42)



def sax(data, win_size, alphabetSize, k,label):

    rows, cols = data.shape
    print(rows, cols)


    # zscore
    data_norm = sax_knn.zscore(data)

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    bit_data = list()


    for d in data_norm:
        data_paa = sax_knn.PAA(d, win_size)
        data_paa_inv = sax_knn.paa_inv(data_paa, win_size)

        bit_tmp = ''
        paa_tmp = list()


        for i in range(len(d)):
            if d[i] < data_paa_inv[i]:
                bit_tmp += '0'
            else:
                bit_tmp += '1'
        # print(bit_tmxp)
        # print(len(bit_tmp))
        bit_data.append(int(bit_tmp, 2))

        # paa2letter
        alpha = sax_knn.paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

    paa_trans = np.array(paa_trans)
    paa = np.array(paa)
    paa_alpha = np.array(paa_alpha)

    # compare the two series
    raw_dist = np.zeros((rows, rows))
    # print(paa_alpha)
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # print(paa_alpha[i])
            # print(paa_alpha[j])
            raw_dist[i, j] = sax_knn.compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])

            ###calculate bit distance
            c = bit_data[i] ^ bit_data[j]
            ones = 0
            while c:
                ones += 1
                c &= (c - 1)

            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]

            raw_dist[i, j] += np.sqrt(ones * 1.0 / win_size)  # bit dist

            raw_dist[j, i] = raw_dist[i, j]

        arg_index = np.argsort(raw_dist[i])
        tmp = dict()
        if i in arg_index[:k]:
            for l in arg_index[:k + 1]:
                if l == i:
                    continue
                tmp[label[l]] = tmp.get(label[l], 0) + 1
        else:
            for l in arg_index[:k]:
                tmp[label[l]] = tmp.get(label[l], 0) + 1

        pre = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        pred.append(pre)
        # print(pre)

    preanomaly = list()
    trueanomaly = list()
    for i in range(rows):
        if label[i] != 0:
            label[i] = 1
            trueanomaly.append(i)

        if pred[i] != 0:
            pred[i] = 1
            preanomaly.append(i)

    print(preanomaly)
    print(trueanomaly)

    print('win_size: %d, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
          (win_size, precision_score(label, pred, average='macro'),
           recall_score(label, pred, average='macro'),
           f1_score(label, pred, average='macro'),
           roc_auc_score(label, pred, ),
           mean_squared_error(label, pred),
           ))
    # print('auc: %f, error rate: %f', (roc_auc_score(label, pred, ), mean_squared_error(label, pred)))



    return raw_dist, preanomaly, trueanomaly

if __name__ == '__main__':
    df1 = pd.read_csv('../data/others/mitbih_test.csv', header=None)
    df2 = pd.read_csv('../data/others/mitbih_train.csv', header=None)
    df = pd.concat([df1, df2], axis=0)

    # print(df.head())
    # print(df.info())

    # 187列是label
    df[187].value_counts()  # 计算series里面相同数据出现的频率

    # 设定 0 为 Normal
    print(df[187].value_counts())



    ECG = df.values  # 查看series的值
    data = ECG[:, :-1]
    labels = ECG[:, -1].astype(int)
    # print(data[1])
    print(labels)

    # 返回非0的数组元组的索引，其中y是要索引数组的条件
    C0 = np.argwhere(labels == 0).flatten()
    C1 = np.argwhere(labels == 1).flatten()
    C2 = np.argwhere(labels == 2).flatten()
    C3 = np.argwhere(labels == 3).flatten()
    C4 = np.argwhere(labels == 4).flatten()

    # newlabel =labels[C0, :]
    # print(C0,C1)
    # exit()

    win_size = 5
    alphabetSize = 5
    k = 3
    _, preanomaly, trueanomaly = sax(data[94250:94850], win_size, alphabetSize, k,labels[94250:94850])