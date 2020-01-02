# -*- coding: utf-8 -*-
# @Time    : 2018/6/13 15:01
# @Author  : Inkky
# @Email   : yingyang_chen@163.com

'''
ref:
https://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html
1.sax
2.sax-td: difference between begin point(end points) and average
3.extended sax: max and min points represented by symbol
4.tsax: trend( our method)
'''

import operator
import time
import sys
import matplotlib
from traditional.SAX.original_sax import zscore, toPAA, paa_inv, paa2letter,compareTS
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix


# from PAA.saxecg import toPAA, paa_inv, paa2letter, zscore, compareTS


class StringsAreDifferentLength(Exception): pass


def original_sax(data, win_size, alphabetSize, k, label):
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()

    for d in data_norm:  # d 一条数据
        data_paa = toPAA(d, win_size)  # 记录均值
        data_paa_inv = paa_inv(data_paa, win_size)
        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

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

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            # raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist
            raw_dist[i, j] = np.sqrt(rows/win_size) * raw_dist[i, j]  # sax dist
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

    return raw_dist, pred


def sax_sd(data, win_size, alphabetSize, k, label):
    '''
    GET the standard deviation of each segment
    :return:
    '''
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    trainBeginTime = time.clock()
    sd = list()  # sax_sd 存放均方差的值

    for d in data_norm:  # d 一条数据
        data_paa = toPAA(d, win_size)  # 记录均值
        data_paa_inv = paa_inv(data_paa, win_size)
        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

        ########## SAX_SD ##############
        # GET the standard deviation of each segment
        i = 0
        tmpsd = list()
        while i < len(d) / win_size:
            tmpsd.append(np.sqrt(np.var(d[i * win_size:(i + 1) * win_size])))
            i = i + 1

        sd.append(tmpsd)
        sd = np.array(sd)

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

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            ##### sax_sd distance ########
            raw_dist[i, j] += np.sqrt(np.sum(np.square(sd[i] - sd[j])))

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
    return raw_dist, pred


def esax(data, win_size, alphabetSize, k, label):
    '''
    max and min points represented by symbol
    :return:
    '''
    # print('esax')
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    trainBeginTime = time.clock()
    tmax = list()  # extended sax 存放每段max的值
    tmin = list()  # extended sax 存放每段min的值

    for d in data_norm:  # d 一条数据
        data_paa = toPAA(d, win_size)  # 记录均值
        data_paa_inv = paa_inv(data_paa, win_size)
        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

        ########## Extended SAX ##########
        # max and min points
        i = 0
        tmpmax = list()
        tmpmin = list()
        while i < (len(d) / win_size):
            tmpmax.append(max(d[i * win_size:(i + 1) * win_size]))
            tmpmin.append(min(d[i * win_size:(i + 1) * win_size]))
            i = i + 1

        # paa2letter
        alphamax = paa2letter(tmpmax, alphabetSize)
        alphamin = paa2letter(tmpmin, alphabetSize)
        # print('alphamax',alphamax,len(alphamax))
        # print('alphamin', alphamin,len(alphamin))
        tmax.append(alphamax)
        tmin.append(alphamin)

    tmax = np.array(tmax)
    tmin = np.array(tmin)
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

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            #### extended sax distance ########
            # raw_dist[i, j] += np.sqrt(np.sum(np.square(tmax[i] - tmax[j]) + np.square(tmin[i] - tmin[j])))
            raw_dist[i, j] += np.sqrt(win_size) * (
                        compareTS(tmax[i], tmax[j], alphabetSize) + compareTS(tmin[i], tmin[j], alphabetSize))

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
    return raw_dist, pred


def sax_td(data, win_size, alphabetSize, k, label):
    '''
    save the difference between begin/end time point value and mean value
    :return:
    '''
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    bit_data = list()
    trainBeginTime = time.clock()
    ts = list()  # SAX_TD 存放starting point 和mean的差值
    te = list()  # SAX_TD 存放ending point 和mean的差值
    tmax = list()  # extended sax 存放每段max的值
    tmin = list()  # extended sax 存放每段min的值
    sd = list()  # sax_sd 存放均方差的值
    # print('1')

    for d in data_norm:  # d 一条数据
        data_paa = toPAA(d, win_size)  # 记录均值
        data_paa_inv = paa_inv(data_paa, win_size)
        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

        # ########## SAX_TD ###################
        # save the begin and end time point value
        i = 0
        tmpts = list()
        tmpte = list()
        # print(len(d))
        # print(win_size)
        while i < len(d) / win_size:
            # print(i)
            tmpts.append(d[i * win_size] - data_paa[i])
            tmpte.append(d[min((i + 1) * win_size, len(d)) - 1] - data_paa[i])
            i = i + 1

        # print(tmpts)
        # print(tmpte)
        # print(len(tmpte))
        ts.append(tmpts)
        te.append(tmpte)

    paa_trans = np.array(paa_trans)
    paa = np.array(paa)
    paa_alpha = np.array(paa_alpha)
    ts = np.array(ts)
    te = np.array(te)
    sd = np.array(sd)
    tmax = np.array(tmax)
    tmin = np.array(tmin)

    # compare the two series
    raw_dist = np.zeros((rows, rows))
    # print(paa_alpha)
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # print(paa_alpha[i])
            # print(paa_alpha[j])

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            ##### sax_td distance #####
            raw_dist[i, j] += np.sqrt(np.sum(np.square(ts[i] - ts[j]) + np.square(te[i] - te[j])))  # sax_td dist

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
    return raw_dist, pred


def tsax(data, win_size, alphabetSize, k, label):
    '''
    save the relative binary trend
    :return:
    '''
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    bit_data = list()
    trainBeginTime = time.clock()

    for d in data_norm:  # d 一条数据
        data_paa = toPAA(d, win_size)  # 记录均值
        data_paa_inv = paa_inv(data_paa, win_size)
        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

        # print(data_paa_inv)
        print(paa_alpha)



        ############# TSAX ##############
        # save the relative binary trend
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

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            ########### calculate bit distance ##########
            c = bit_data[i] ^ bit_data[j]
            ones = 0
            while c:
                ones += 1
                c &= (c - 1)

            raw_dist[i, j] += np.sqrt(ones * 1.0 / win_size)  # bit dist

            raw_dist[j, i] = raw_dist[i, j]
        arg_index = np.argsort(raw_dist[i]) #数组值从小到大的索引值
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
        # print(pre)
        # exit()
        pred.append(pre)
    return raw_dist, pred


def notify_result(method_name, data, win_size, alphabetSize, k, label):
    numbers = {
        tsax: tsax,
        sax_td: sax_td,
        esax: esax,
        sax_sd: sax_sd,
        original_sax: original_sax
    }

    method = numbers.get(method_name)
    # print(method_name)
    if method:
        disti, prediction = method(data, win_size, alphabetSize, k, label)
        return disti, prediction



if __name__ == '__main__':

    # parameters
    win_size = 4
    alphabetSize = 3
    k = 3

    data = np.loadtxt('../data/UCR(TRAIN+TEST)/BeetleFly.txt', delimiter=',')
    label = data[:, 0]
    data = data[:, 1:]
    rows, cols = data.shape

    trainBeginTime = time.clock()
    raw_dist, pred = notify_result(original_sax, data, win_size, alphabetSize, k, label)
    print(raw_dist)
    trainEndTime = time.clock()

    preanomaly = list()
    trueanomaly = list()
    for i in range(rows):
        if label[i] != 1.0:
            label[i] = 0
            trueanomaly.append(i)

        if pred[i] != 1.0:
            pred[i] = 0
            preanomaly.append(i)

    # print which time point is anomaly
    # print(len(preanomaly))
    # print(len(trueanomaly))

    # result
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    # print('TN: %d,tp: %d,fn: %d,FP: %d', tn, tp, fn, fp)
    specificity = tn / (tn + fp)
    falseAlarmRate = fp / (fp + tn)
    #
    # print('win_size: %d, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f,time:  %f' % \
    #       (win_size, precision_score(label, pred, average='macro'),
    #        recall_score(label, pred, average='macro'),
    #        f1_score(label, pred, average='macro'),
    #        roc_auc_score(label, pred, ),
    #        mean_squared_error(label, pred),
    #        (trainEndTime - trainBeginTime)))

    # print('specificity: ', specificity, ',false alarm rate: %f', falseAlarmRate)
    # print('auc:%f, error rate: %f'%\
    #       (roc_auc_score(label, pred, ), mean_squared_error(label, pred)))
