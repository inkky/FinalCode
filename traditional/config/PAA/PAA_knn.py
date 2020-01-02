# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 14:26
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''

'''
import os
import sys
import numpy
import operator
import random
from numpy import array, sum, sqrt
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
from matplotlib import pyplot as plt


def R_PAA(data):
    # f = open('new/result_mean_mean_w.txt', 'a')
    # data = numpy.loadtxt('sorted/' + filename, delimiter=',')
    # print(filename)
    labels = data[:, 0]
    data = data[:, 1:]
    rows, cols = data.shape

    win = 0
    precision = 0
    recall = 0
    f1 = 0
    auc = 0
    error_rate=1
    t=0
    f1_list = list()
    error_list = list()
    print('paa')

    for i in range(2, 11):
        win_size = i
        k = 3
        paa_data = list()
        paa_up = list()
        paa_below = list()
        # up_numlist = list()
        # be_numlist = list()
        paa_var = list()
        trainBeginTime = time.clock()
        for d in data:
            paa_tmp = list()
            up = list()  # 一条时间序列所有up的集合
            below = list()
            up_tmp = list()  # 每段up的集合
            below_tmp = list()
            # num_uptmp = list()
            # num_betmp = list()
            var_tmp = list()
            for i in range(0, len(d), win_size):
                low = i
                up_num = 0  # 每段up的个数
                below_num = 0
                high = min(len(d), i + win_size)
                PAA = sum(d[low: high]) / (high - low)  # 算均值mean
                # VAR = sqrt(sum(numpy.square(d[low:high] - PAA)) / (high - low))

                for j in range(low, high):  # 分段算大于均值的部分的均值
                    if d[i] > PAA:
                        up_tmp.append(d[i] - PAA)
                        up_num += 1
                    else:
                        below_num += 1
                        below_tmp.append(PAA - d[i])
                paa_tmp.append(PAA)
                # num_uptmp.append(up_num)
                # num_betmp.append(below_num)
                # var_tmp.append(VAR)

                if up_num == 0:
                    mean_up = 0
                else:
                    mean_up = sum(up_tmp) / up_num
                if below_num == 0:
                    mean_below = 0
                else:
                    mean_below = sum(below_tmp) / below_num

                up.append(mean_up)
                below.append(mean_below)
            paa_data.append(paa_tmp)
            paa_var.append(var_tmp)
            paa_up.append(up)  # 所有时间序列的up的集合
            paa_below.append(below)
            # up_numlist.append(num_uptmp)
            # be_numlist.append(num_betmp)

        paa_data = numpy.array(paa_data)
        # paa_var = numpy.array(paa_var)
        paa_below = numpy.array(paa_below)
        paa_up = numpy.array(paa_up)
        # up_numlist = numpy.array(up_numlist)
        # be_numlist = numpy.array(be_numlist)

        raw_dist = numpy.zeros((rows, rows))
        # eu_dist = numpy.zeros((rows, rows))
        # tightness = list()

        pred = list()
        for i in range(rows):
            for j in range(i + 1, rows):
                tight = list()
                # Euclidean Distance
                # eu_dist[i, j] = numpy.sum(numpy.square(data[i] - data[j]))
                # eu_dist[i, j] **= 0.5

                # PAA Distance
                # raw_dist[i, j] = numpy.linalg.norm(paa_data[i] - paa_data[j])
                raw_dist[i, j] = numpy.sum(numpy.square(paa_data[i] - paa_data[j])) * win_size
                raw_dist[i, j] *= win_size

                # 计算高于均值的差值的均值 result_mean.txt
                # raw_dist[i, j] += numpy.sum(up_numlist[i]*
                #     numpy.square(paa_up[i] - paa_up[j]))+ sum(be_numlist*numpy.square(paa_below[i] - paa_below[j]))

                # 计算高于均值的时间序列的均值 result_mean_mean.txt
                raw_dist[i, j] += numpy.sum(
                    numpy.square((paa_up[i] + paa_data[i]) - (paa_up[j] + paa_data[j])) + numpy.square(
                        (paa_data[i] - paa_below[i]) - (paa_data[j] - paa_below[j])))*(win_size/2)

                # LB_DS,加入方差 result_LS_DS.txt
                # raw_dist[i, j] += numpy.sum(numpy.square(paa_var[i] - paa_var[j])) * win_size

                raw_dist[i, j] **= 0.5

                # print(raw_dist[i, j], eu_dist[i, j])
                # tight.append(raw_dist[i, j] / eu_dist[i, j])

                # cosin
                # cos1=numpy.sum(paa_data[i] * paa_data[j])
                # cos21=sum(numpy.square(paa_data[i]))**0.5
                # cos22=sum(numpy.square(paa_data[j]))**0.5
                # raw_dist[i, j] = 1-cos1/float(cos21*cos22)
                # print(raw_dist[i, j])
                # exit()

                raw_dist[j, i] = raw_dist[i, j]

            arg_index = numpy.argsort(raw_dist[i])
            tmp = dict()
            if i in arg_index[:k]:
                for l in arg_index[:k + 1]:
                    if l == i:
                        continue
                    tmp[labels[l]] = tmp.get(labels[l], 0) + 1
            else:
                for l in arg_index[:k]:
                    tmp[labels[l]] = tmp.get(labels[l], 0) + 1

            pre = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            pred.append(pre)

        # tightness.append(numpy.mean(tight))
        # print(tightness)
        # exit()

        for i in range(rows):
            if labels[i] != 1.0:
                labels[i] = 0

            if pred[i] != 1.0:
                pred[i] = 0

        print('win_size: %d, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f,time:  %f' % \
              (win_size, precision_score(labels, pred, average='macro'),
               recall_score(labels, pred, average='macro'),
               f1_score(labels, pred, average='macro'),
               roc_auc_score(labels, pred, ),
               mean_squared_error(labels,pred),
               (time.clock() - trainBeginTime)))
        # print('COST TIME     : %f' % (time.clock() - trainBeginTime))

        # t_precision = precision_score(labels, pred, average='macro')
        # t_error_rate=mean_squared_error(labels,pred)
        # t_f1 = f1_score(labels, pred, average='macro')
        # if (t_error_rate < error_rate):
        #     # f1 = t_f1
        #     error_rate=t_error_rate
        #     win = win_size
        #     f1 = f1_score(labels, pred, average='macro')
        #     auc = roc_auc_score(labels, pred)
        #     recall = recall_score(labels, pred, average='macro')
        #     precision = precision_score(labels, pred, average='macro')
            # t = time.clock() - trainBeginTime

        # f1_list.append(time.clock() - trainBeginTime)
        error_list.append(mean_squared_error(labels, pred))
    # print(f1_list)
    # exit()

    # f.write(filename)
    # f.write(',%d,%f,%f,%f,%f,%f' % \
    #         (win, precision, recall, f1, auc, error_rate))
    # f.write('\n')
    # f.close()

    return f1_list,error_list


# filedir = os.getcwd() + '/sorted'
# # filename = 'ecg200.txt'
# # PAA(filename)
# # print(tmp)
# for filename in os.listdir(filedir):
#     # f = open('result/knn_cosin_error.txt', 'a')
#     print(filename)
#     data = numpy.loadtxt('sorted/' + filename, delimiter=',')
#     R_PAA(data)
#     # print(timelist)
