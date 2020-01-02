# !/usr/bin/env python
# -*-coding:utf-8-*-
import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from traditional.PAA.expand_PAA import *
import time


def find_kdist(d_matrix, k):
    rows, cols = d_matrix.shape
    d_ndist = numpy.arange(rows)
    d_n = numpy.zeros((rows, k), dtype='int')
    for r in range(rows):
        if r > k:
            d_n[r] = numpy.arange(k)
        else:
            d_n[r] = numpy.append(numpy.arange(r), numpy.arange(r + 1, k + 1))
        tmp = sorted(d_matrix[r, d_n[r]])
        for c in range(cols):
            if r == c:
                continue
            i = k - 2
            while i >= 0 and d_matrix[r, c] < tmp[i]:
                tmp[i + 1] = tmp[i]
                d_n[r, i + 1] = d_n[r, i]
                i -= 1
            if i != k - 2:
                d_n[r, i + 1] = c
                tmp[i + 1] = d_matrix[r, c]
        d_ndist[r] = tmp[-1]
    return d_ndist, d_n


def reach_distance(d_nlist, d_matrix):
    rows = d_nlist.shape[0]
    rd_matrix = numpy.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i == j:
                continue
            else:
                rd_matrix[i, j] = max(d_nlist[j], d_matrix[i, j])
    return rd_matrix


def lr_density(d_n, rd_matrix):
    rows = d_n.shape[0]
    lrd = numpy.zeros(rows)
    for i in range(rows):
        lrd[i] = len(d_n[i]) * 1.0 / sum(rd_matrix[i, d_n[i]])
    return lrd


def lo_factor(d_n, lr_vector):
    rows, cols = d_n.shape
    lof = numpy.zeros(rows)
    for i in range(rows):
        lof[i] = sum(lr_vector[d_n[i]]) * 1.0 / (cols * lr_vector[i])
    return lof


if __name__ == '__main__':
    # X = [[2.5, 3.4], [6, 8]]
    # Y=[1,0]
    # data = numpy.loadtxt('twoclass/Computers.txt', delimiter=',')
    # labels = data[:, 0]
    # print(labels)
    # dist_matrix=euclidean_distances(data[:, 1:])
    # dist_matrix = numpy.array(dist_matrix)
    # d_nlist, d_n = find_kdist(dist_matrix,k=5)
    # rd_matrix = reach_distance(d_nlist, dist_matrix)
    # lr_vector = lr_density(d_n, rd_matrix)
    # lof = lo_factor(d_n, lr_vector)
    # auc = roc_auc_score(labels, lof)
    # print(auc)
    # exit()

    filedir = os.getcwd() + '/../data/UCRtwoclass/'
    for filename in os.listdir(filedir):
        # print(filename)
        data = numpy.loadtxt(filedir + filename, delimiter=',')
        labels = data[:, 0]
        # f = open('result/time_lof_Cos_k.txt', 'a')
        max_auc = 0

        #     for k in range(5,7):
        #         ##Euclidean
        #         dist_matrix = euclidean_distances(data[:, 1:])
        #         dist_matrix = numpy.array(dist_matrix)
        #         d_nlist, d_n = find_kdist(dist_matrix,k)
        #         rd_matrix = reach_distance(d_nlist, dist_matrix)
        #         lr_vector = lr_density(d_n, rd_matrix)
        #         lof = lo_factor(d_n, lr_vector)
        #         auc = roc_auc_score(labels, lof)
        #
        #         if auc < 0.5:
        #             auc = 1 - auc
        #         print(auc)
        #
        #         if max_auc < auc:
        #             max_auc = auc
        #     f.write(filename)
        #     f.write(',%f' % max_auc)
        #     f.write('\n')
        # f.close()

        # ### test
        # X = [[2.5, 3.4], [6, 8]]
        # Y=[1,0]
        # dist_matrix=euclidean_distances(X)
        # dist_matrix = numpy.array(dist_matrix)
        # d_nlist, d_n = find_kdist(dist_matrix,k=1)
        # rd_matrix = reach_distance(d_nlist, dist_matrix)
        # lr_vector = lr_density(d_n, rd_matrix)
        # lof = lo_factor(d_n, lr_vector)
        # auc = roc_auc_score(Y, lof)
        # print(auc)
        # exit()

        # dist_matrix = euclidean_distances(data[:, 1:])
        # dist_matrix = euclidean_distances(data)
        # dist_matrix=BT_PAA(3,data)
        # dist_matrix=Raw_PAA(7,data)
        # dist_matrix = numpy.array(dist_matrix)
        # print(dist_matrix)
        # exit()

        beginTime = time.clock()
        for k in range(3, 4):
            for win_size in range(3, 20):

                # different methods
                # dist_matrix,_ = Raw_PAA(win_size, data)
                # dist_matrix,_ = NT_PAA(win_size, data)
                # dist_matrix,_ = Cosine_PAA(win_size, data)
                dist_matrix, _ = BT_PAA(win_size, data,labels,k)

                # LOF
                d_nlist, d_n = find_kdist(dist_matrix, k)
                rd_matrix = reach_distance(d_nlist, dist_matrix)
                lr_vector = lr_density(d_n, rd_matrix)
                lof = lo_factor(d_n, lr_vector)


                # metrics
                auc = roc_auc_score(labels, lof)


                if auc < 0.5:
                    auc = 1 - auc

                if max_auc < auc:
                    max_auc = auc
                    best_win = win_size
        # print('data:%s, win_size: %d, auc: %.4f, Precision: %f, Recall: %f, F1: %f' % (
        # filename, best_win, max_auc, precision_score(labels, lof, average='macro'),
        # recall_score(labels, lof, average='macro'),
        # f1_score(labels, lof, average='macro')))


        # f.write(filename)
        # f.write(',%f,%f' % max_auc)
        # f.write('\n')

# f.close()

# dist_matrix = numpy.array(dist_matrix)
#         d_nlist, d_n = find_kdist(dist_matrix, k=d)
#         rd_matrix = reach_distance(d_nlist, dist_matrix)
#         lr_vector = lr_density(d_n, rd_matrix)
#         lof = lo_factor(d_n, lr_vector)
#         end = time.clock()
#         ruc = roc_auc_score(label, lof)
