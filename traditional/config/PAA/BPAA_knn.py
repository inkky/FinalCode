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
from numpy import array, sum, sqrt
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from traditional.PAA.expand_PAA import BT_PAA, Raw_PAA, NT_PAA, Cosine_PAA


def data_scale(data):
    """归一化"""
    print('------------data_scale---------------')
    standscaler = StandardScaler()
    mscaler = MinMaxScaler(feature_range=(0, 1))
    data = standscaler.fit_transform(data)
    data = mscaler.fit_transform(data)
    return data

def B_PAA(data):

    labels = data[:, 0]
    data = data[:, 1:]
    rows, cols = data.shape
    print(data.shape)
    data=data_scale(data)

    win = 0
    precision = 0
    recall = 0
    f1 = 0
    auc = 0
    f1_list = list()
    print('bpaa')
    error_rate=1
    error_list = list()
    time_list=list()


    for i in range(4, 26,2):
        win_size = i
        k = 3

        bit_data = list()
        paa_data = list()

        trainBeginTime = time.clock()
        for d in data:
            tmp = list()
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
                # ones = 0
                # while c:
                #     ones += 1
                #     c &= (c - 1)

                ones = bin(bit_data[i] ^ bit_data[j]).count('1')
                raw_dist[i, j] += ones * 1.0 / win_size

                # 计算max之间的距离
                gap = 0
                if ones > 1:
                    # print(ones)
                    tmp = [k for k, v in enumerate(c) if v == '1']
                    gap = max([tmp[k + 1] - tmp[k] for k in range(ones - 1)])
                    # print(gap)

                raw_dist[i, j] += gap /cols*win_size  # 处理gap到（0,1）

                # 计算高于均值的时间序列的均值 result_mean_mean.txt
                # raw_dist[i, j] += numpy.sum(
                #     numpy.square((paa_up[i] + paa_data[i]) - (paa_up[j] + paa_data[j])) + numpy.square(
                #         (paa_data[i] - paa_below[i]) - (paa_data[j] - paa_below[j]))) * (win_size / 2)


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

        for i in range(rows):
            if labels[i] != 1:
                labels[i] = 0

            if pred[i] != 1.0:
                pred[i] = 0

        print('win_size: %d, Precision: %f, Recall: %f, F1: %f,auc:  %f, error:  %f , time:  %f' % \
              (win_size, precision_score(labels, pred, average='macro'),
               recall_score(labels, pred, average='macro'),
               f1_score(labels, pred, average='macro'),
               roc_auc_score(labels, pred ),
               mean_squared_error(labels, pred),
               (time.clock() - trainBeginTime)))
        # print('COST TIME     : %f' % (time.clock() - trainBeginTime))

        # t_error_rate = mean_squared_error(labels, pred)
        # t_f1 = f1_score(labels, pred, average='macro')
        # if (t_error_rate < error_rate):
        #     # f1 = t_f1
        #     error_rate = t_error_rate
        #     win = win_size
        #     f1 = f1_score(labels, pred, average='macro')
        #     auc = roc_auc_score(labels, pred)
        #     recall = recall_score(labels, pred, average='macro')
        #     precision = precision_score(labels, pred, average='macro')
            # t = time.clock() - trainBeginTime
        # f.write(filename)
        # f.write(',%f' % (time.clock() - trainBeginTime))
        # f.write('\n')

        # f1_list.append(t)
        error_list.append(mean_squared_error(labels, pred))
        time_list.append(time.clock() - trainBeginTime)
        # print(error_list)
        # print(raw_dist)
        # exit()

        # print(win_size,numpy.mean(raw_dist))

    # f.write(filename)
    # f.write(',%d,%f,%f,%f,%f,%f' % \
    #         (win, precision, recall, f1, auc, error_rate))
    # f.write('\n')
    # f.close()


    return raw_dist,error_list,time_list



# plt.figure()
# fig = plt.gcf()
# fig.set_size_inches(8, 4)
# plt.xlim((0, 20))
# plt.ylabel('cost time')
# plt.xlabel('window_size')
#
# # x=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

filedir = os.getcwd() + '/error'
# for filename in os.listdir(filedir):
#     # f = open('result/knn_bpaa_error.txt', 'a')
#     data = numpy.loadtxt('error/' + filename, delimiter=',')
#     print(filename)
#     bpaa_list,error_list = B_PAA(data)
#     print(error_list)
#     plt.figure()
#     plt.plot(error_list,'o',label='Computers')
#     plt.show()
    # exit()
    # paa_list=PAA(filename)
    # x = range(2, 20, 1)
    # plt.plot(x, bpaa_list, 'o-', label='BPAA_'+filename)
    # plt.plot(x, paa_list, '*-', label='PAA_'+filename)
# plt.legend(loc='best')
# plt.xticks(x)
# plt.tight_layout()
# # plt.savefig('img/costtime.png',dpi=300)
# plt.show()

'''
coffee:加上gap距离和paa距离的效果好
computer：只有趋势距离最好
ECG：三个距离一起最好
Gun_Point.txt：三个一起最好
Ham.txt：三个一起最好

'''
filename = '../data/UCRtwoclass/Earthquakes.txt'
data = numpy.loadtxt(filename, delimiter=',')

B_PAA(data)
