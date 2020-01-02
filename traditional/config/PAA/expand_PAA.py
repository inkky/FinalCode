# @Time    : 2018/6/1 10:20
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''
w 滑动窗口的个数
'''

# 欧氏距离
def EU( data,labels,k):
    rows, cols = data.shape

    k = 3
    trainBeginTime = time.clock()
    raw_dist = numpy.zeros((rows, rows))
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # raw_dist[i, j] = numpy.linalg.norm(data[i] - data[j])
            raw_dist[i, j] = numpy.sum(numpy.square(data[i] - data[j]))
            raw_dist[i, j] **= 0.5
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
    return raw_dist,pred

'''
trend+NT
'''
def TPAA(win_size, data,labels,k):
    rows, cols = data.shape
    bit_data = list()
    paa_data = list()
    paa_up = list()
    paa_below = list()

    trainBeginTime = time.clock()
    print('representation')
    for d in data:
        bit_tmp = ''
        paa_tmp = list()
        up = list()  # 一条时间序列所有up的集合
        below = list()
        for i in range(0, len(d), win_size):
            low = i
            up_num = 0  # 每段up的个数
            below_num = 0
            mean_up = 0
            mean_below = 0

            high = min(len(d), i + win_size)
            PAA = sum(d[low: high]) / (high - low)
            paa_tmp.append(PAA)

            #NT, up and below
            for j in range(low, high):  # 分段算大于均值的部分的均值
                if d[i] > PAA:
                    mean_up += d[i] - PAA
                    up_num += 1
                else:
                    below_num += 1
                    mean_below += PAA - d[i]

            if up_num != 0:
                mean_up = mean_up / up_num
            if below_num != 0:
                mean_below = mean_below / below_num

            up.append(mean_up)
            below.append(mean_below)

            # bit array
            for j in range(low, high):
                if d[j] < PAA:
                    bit_tmp += '0'
                else:
                    bit_tmp += '1'
        bit_data.append(int(bit_tmp, 2))
        # bit_data.append(bit_tmp)
        paa_data.append(paa_tmp)
        paa_up.append(up)  # 所有时间序列的up的集合
        paa_below.append(below)

    paa_data = numpy.array(paa_data)
    paa_below = numpy.array(paa_below)
    paa_up = numpy.array(paa_up)

    raw_dist = numpy.zeros((rows, rows))

    print('calculate the distance')
    pred = list()
    for i in range(rows):
        # print(i)
        for j in range(i + 1, rows):
            mscaler = MinMaxScaler(feature_range=(0, 1))

            # print('PAA distance')
            raw_dist[i, j] = numpy.linalg.norm(paa_data[i] - paa_data[j])
            # raw_dist[i, j] = numpy.sum(numpy.square(paa_data[i] - paa_data[j]))
            raw_dist[i, j] *= win_size

            # print('mean distance')
            # 计算高于均值的时间序列的均值 result_mean_mean.txt
            raw_dist[i, j] +=(numpy.sum(
                numpy.square((paa_up[i]) - (paa_up[j])) + numpy.square(
                    (paa_below[i]) - (paa_below[j]))))/win_size


            # print('bit distance')
            #计算bit array的距离
            raw_tmp = numpy.zeros((rows, rows))
            c = bin(bit_data[i] ^ bit_data[j])
            ones = bin(bit_data[i] ^ bit_data[j]).count('1')
            # raw_dist[i, j] += ones * 1.0 / win_size
            raw_dist[i, j] += ones * 1.0
            # raw_tmp[i, j] += ones * 1.0*win_size


            # 计算max最大波动间距
            raw_tmp = numpy.zeros((rows, rows))
            gap = 0
            if ones > 1:
                # print(ones)
                tmp = [k for k, v in enumerate(c) if v == '1']
                gap = max([tmp[k + 1] - tmp[k] for k in range(ones - 1)])
                # print(gap)

            # raw_tmp[i, j] += (gap / cols) * win_size  # 处理gap到（0,1）
            raw_tmp[i, j] += (gap / cols)   # 处理gap到（0,1）
            # raw_tmp[i, j] += (gap * win_size)   # 处理gap到（0,1）


            # raw_tmp = mscaler.fit_transform(raw_tmp)
            raw_dist += raw_tmp

            # 计算

            raw_dist[i, j] **= 0.5
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
    return raw_dist, pred

# paa+binary trend
def BT_PAA(win_size, data,labels,k):
    rows, cols = data.shape
    bit_data = list()
    paa_data = list()

    trainBeginTime = time.clock()
    for d in data:
        bit_tmp = ''
        paa_tmp = list()
        for i in range(0, len(d), win_size):
            low = i
            high = min(len(d), i + win_size)
            PAA = sum(d[low: high]) / (high - low)
            paa_tmp.append(PAA)

            for j in range(low, high):
                if d[j] < PAA:
                    bit_tmp += '0'
                else:
                    bit_tmp += '1'
        bit_data.append(int(bit_tmp, 2))
        # bit_data.append(bit_tmp)
        paa_data.append(paa_tmp)
    paa_data = numpy.array(paa_data)
    raw_dist = numpy.zeros((rows, rows))

    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # raw_dist[i, j] = numpy.linalg.norm(paa_data[i] - paa_data[j])
            raw_dist[i, j] = numpy.sum(numpy.square(paa_data[i] - paa_data[j]))
            raw_dist[i, j] *= win_size

            c = bin(bit_data[i] ^ bit_data[j])
            ones = bin(bit_data[i] ^ bit_data[j]).count('1')
            # raw_dist[i, j] += ones * 1.0 / win_size
            raw_dist[i, j] += ones * 1.0


            # 计算max之间的距离
            gap = 0
            if ones > 1:
                # print(ones)
                tmp = [k for k, v in enumerate(c) if v == '1']
                gap = max([tmp[k + 1] - tmp[k] for k in range(ones - 1)])
                # print(gap)

            # raw_dist[i, j] += gap / cols * win_size  # 处理gap到（0,1）

            #计算



            raw_dist[i, j] **= 0.5
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
    return raw_dist,pred

# 原始的paa
def Raw_PAA(win_size, data,labels,k):
    rows, cols = data.shape

    paa_data = list()
    trainBeginTime = time.clock()
    for d in data:
        paa_tmp = list()
        for i in range(0, len(d), win_size):
            low = i
            high = min(len(d), i + win_size)
            PAA = sum(d[low: high]) / (high - low)
            paa_tmp.append(PAA)
        paa_data.append(paa_tmp)
    paa_data = numpy.array(paa_data)
    raw_dist = numpy.zeros((rows, rows))
    pred=list()
    for i in range(rows):
        for j in range(i + 1, rows):
            raw_dist[i, j] = numpy.sum(numpy.square(paa_data[i] - paa_data[j]))
            raw_dist[i, j] *= win_size
            raw_dist[i, j] **= 0.5
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
    return raw_dist,pred

# paa+差均值
def NT_PAA(win_size, data,labels,k):
    rows, cols = data.shape
    paa_data = list()
    paa_up = list()
    paa_below = list()
    trainBeginTime = time.clock()
    for d in data:
        paa_tmp = list()
        up = list()  # 一条时间序列所有up的集合
        below = list()
        for i in range(0, len(d), win_size):
            low = i
            up_num = 0  # 每段up的个数
            below_num = 0
            mean_up = 0
            mean_below = 0
            high = min(len(d), i + win_size)
            PAA = sum(d[low: high]) / (high - low)  # 算均值mean

            for j in range(low, high):  # 分段算大于均值的部分的均值
                if d[i] > PAA:
                    mean_up += d[i] - PAA
                    up_num += 1
                else:
                    below_num += 1
                    mean_below += PAA - d[i]
            paa_tmp.append(PAA)

            if up_num != 0:
                mean_up = mean_up / up_num
            if below_num != 0:
                mean_below = mean_below / below_num

            up.append(mean_up)
            below.append(mean_below)
        paa_data.append(paa_tmp)
        paa_up.append(up)  # 所有时间序列的up的集合
        paa_below.append(below)

    paa_data = numpy.array(paa_data)
    paa_below = numpy.array(paa_below)
    paa_up = numpy.array(paa_up)

    raw_dist = numpy.zeros((rows, rows))
    pred=list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # PAA Distance
            # raw_dist[i, j] = numpy.linalg.norm(paa_data[i] - paa_data[j])
            raw_dist[i, j] = numpy.sum(numpy.square(paa_data[i] - paa_data[j]))
            raw_dist[i, j] *= win_size

            # 计算高于均值的时间序列的均值 result_mean_mean.txt
            raw_dist[i, j] += numpy.sum(
                numpy.square((paa_up[i] ) - (paa_up[j] )) + numpy.square(
                    (paa_below[i]) - (paa_below[j])))

            raw_dist[i, j] **= 0.5
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
    return raw_dist,pred

#paa+ cosin角度
def Cosine_PAA(win_size, data,labels,k):
    rows, cols = data.shape
    time_list = list()

    paa_data = list()
    trainBeginTime = time.clock()
    for d in data:
        tmp = list()
        paa_tmp = list()
        for i in range(0, len(d), win_size):
            low = i
            high = min(len(d), i + win_size)
            PAA = sum(d[low: high]) / (high - low)
            paa_tmp.append(PAA)
        paa_data.append(paa_tmp)

    paa_data = numpy.array(paa_data)

    raw_dist = numpy.zeros((rows, rows))
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # cosin
            cos1 = numpy.sum(paa_data[i] * paa_data[j])
            cos21 = sum(numpy.square(paa_data[i])) ** 0.5
            cos22 = sum(numpy.square(paa_data[j])) ** 0.5
            raw_dist[i, j] = 1 - cos1 / float(cos21 * cos22)

            # raw_dist[i, j] = numpy.sum(paa_data[i] * paa_data[i]) / (
            # numpy.sqrt(sum(numpy.square(paa_data[i]))) + numpy.sqrt(sum(numpy.square(paa_data[j]))))

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
    return raw_dist,pred

