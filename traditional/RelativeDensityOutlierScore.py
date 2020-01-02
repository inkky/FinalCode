# !/usr/bin/env python
# -*-coding:utf-8-*-

import numpy
import time
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error,roc_curve
from sklearn.preprocessing import MinMaxScaler, scale, minmax_scale
from collections import Counter
from Base import *

def cal_knn_graph(dist_matrix, k):
    rows, _ = dist_matrix.shape
    connection_graph = numpy.zeros((rows, rows))
    for i in range(rows):
        dist_tmp = numpy.argsort(dist_matrix[i])
        for j in range(1, k+1):
            connection_graph[i][dist_tmp[j]] = 1
    return connection_graph


def get_outbound(conn_graph, p):
    rows, _ = conn_graph.shape
    outbounds = list()
    for i in range(rows):
        if conn_graph[p][i] == 1:
            outbounds.append(i)
    return outbounds


def get_inbound(conn_graph, p):
    rows, _ = conn_graph.shape
    inbounds = list()
    for i in range(rows):
        if conn_graph[i][p] == 1:
            inbounds.append(i)
    return inbounds


def get_kernel_density(dist_matrix, n_set, x, h, d):
    tmp = 0
    for n in n_set:
        tmp += 1.0/((2*numpy.pi)**(d/2))*numpy.exp(-((dist_matrix[x][n]**2)/(2*h)))/(h**d)
    tmp += 1.0/((2*numpy.pi)**(d/2))/(h**d)
    return tmp/(len(n_set)+1)


def cal_density_score(dist_matrix, k, h, d):
    conn_graph = cal_knn_graph(dist_matrix, k)
    rows, cols = dist_matrix.shape
    prob = list()
    r_nn = dict()
    for i in range(rows):
        knn = get_outbound(conn_graph, i)
        rnn = get_inbound(conn_graph, i)
        snn = list()
        for x in knn:
            s_rnn = get_inbound(conn_graph, x)
            snn = list(set(snn).union(set(s_rnn)))
        nn = list(set(knn).union(set(rnn)).union(snn))
        r_nn[i] = nn
        p = get_kernel_density(dist_matrix, nn, i, h=h, d=d)
        prob.append(p)

    result = list()
    for i in range(rows):
        l = len(r_nn.get(i))
        tmp = 0
        for n in r_nn.get(i):
            tmp += prob[n]
        result.append(1.0*tmp/(l*prob[i]))
    return result


if __name__ == '__main__':
    dir_path = '../../data/UCRtwoclass/'
    res = open('../../traditionalresult/rdos15_result.txt', 'a')
    for f in os.listdir(dir_path):
        # if f == 'HandOutlines' or f == 'WormsTwoClass':
        #     continue
        # if f != 'DiatomSizeReduction_4_1':
        #     continue
        file = dir_path + f
        data = numpy.loadtxt(file, delimiter=',')
        labels = data[:, 0]
        data = data[:, 1:]
        rows, cols = data.shape
        c = Counter(labels)
        print(c)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))
        labels = numpy.where(labels == c[1][1], 1, 0)

        data = minmax_scale(data, axis=1)

        begin = time.clock()
        # dist_matrix = euclidean_distances(data)
        dist_matrix = numpy.zeros((rows, rows))
        for r in range(rows):
            for c in range(r + 1, rows):
                dist_matrix[r, c] = numpy.power(sum((data[r]-data[c])**2), 0.5)
                dist_matrix[c, r] = dist_matrix[r, c]
                # for d in xrange(cols):
                #     dist_matrix[r, c] += (data[r][d] - data[c][d]) ** 2
                # dist_matrix[r, c] **= 0.5
                # dist_matrix[c, r] = dist_matrix[r, c]

        pred = cal_density_score(dist_matrix, k=15, h=1, d=cols)
        end = time.clock()

        auc = 1 - roc_auc_score(labels, pred)

        # print("Data=%s, AUC=%f, Time=%f" % (f, auc, end - begin))
        # # break
        # res.write("Data=%s, AUC=%f, Time=%f\n" %(f, auc, end-begin))
        # # break
        score_ratio, pre = cal_score_ratio(labels, pred)
        print('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
              (f, precision_score(labels, pre, average='macro'),
               recall_score(labels, pre, average='macro'),
               f1_score(labels, pre, average='macro'),
               roc_auc_score(labels, pre, ),
               mean_squared_error(labels, pre)
               ))
        res.write('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
              (f, precision_score(labels, pre, average='macro'),
               recall_score(labels, pre, average='macro'),
               f1_score(labels, pre, average='macro'),
               roc_auc_score(labels, pre, ),
               mean_squared_error(labels, pre)
               ))




    res.close()
