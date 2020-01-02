#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lenovo on 2019/10/9
import numpy as np
import time
from AL_BLVE.LocalOutlierFactor import *
import heapq


def bit_array(data, win_size):
    rows, cols = data.shape
    bit_data = list()
    paa_data = list()
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
    paa_data = np.array(paa_data)
    # print(paa_data.shape)
    # print(len(bit_data))
    return paa_data,bit_data

def calculate_distance(test_paa,test_bit,rec_paa,rec_bit,win_size,batch_size):
    print('start calculate the next query index...')
    dist_matrix=list()
    for d in range(len(test_bit)):
        raw_dist=np.sum(np.square(test_paa[d]-rec_paa[d]))
        ones=bin(test_bit[d] ^ rec_bit[d]).count('1')
        c = bin(test_bit[d] ^ rec_bit[d])
        raw_dist+=ones*1/win_size

        # 计算最大波动间距
        gap = 0
        if ones > 1:
            # print(ones)
            tmp = [k for k, v in enumerate(c) if v == '1']
            gap = max([tmp[k + 1] - tmp[k] for k in range(ones - 1)])
            # print(gap)

        raw_dist+=gap/len(test_bit)*win_size# 处理gap到（0,1）

        raw_dist**= 0.5
        dist_matrix.append(raw_dist)
    # print(dist_matrix)

    query_idx = list(map(dist_matrix.index, heapq.nsmallest(batch_size, dist_matrix)))
    print(query_idx)
    return query_idx









def bt_paa_lof(win_size, data, batch_size):
    print('start calculate the trend distance')
    rows, cols = data.shape
    bit_data = list()
    paa_data = list()

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
        paa_data.append(paa_tmp)
    paa_data = np.array(paa_data)
    raw_dist = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(i + 1, rows):
            # raw_dist[i, j] = numpy.linalg.norm(paa_data[i] - paa_data[j])
            raw_dist[i, j] = np.sum(np.square(paa_data[i] - paa_data[j]))
            raw_dist[i, j] *= win_size

            c = bin(bit_data[i] ^ bit_data[j])
            ones = bin(bit_data[i] ^ bit_data[j]).count('1')
            raw_dist[i, j] += ones * 1.0 / win_size
            # raw_dist[i, j] += ones * 1.0

            # 计算max之间的距离
            gap = 0
            if ones > 1:
                # print(ones)
                tmp = [k for k, v in enumerate(c) if v == '1']
                gap = max([tmp[k + 1] - tmp[k] for k in range(ones - 1)])
                # print(gap)
            raw_dist[i, j] += gap / cols * win_size  # 处理gap到（0,1）

            raw_dist[i, j] **= 0.5
            raw_dist[j, i] = raw_dist[i, j]

    # LOF
    k = 20
    d_nlist, d_n = find_kdist(raw_dist, k)  # d_n为k距离邻域
    rd_matrix = reach_distance(d_nlist, raw_dist)
    lr_vector = lr_density(d_n, rd_matrix)
    lof = lo_factor(d_n, lr_vector)
    lof = list(lof)

    # lof越小，越正常
    '''选择最接近1的'''
    idx = []
    for i in range(len(lof)):
        if lof[i] > 1:
            idx.append(i)
    query_idx = list(map(lof.index, heapq.nsmallest(batch_size, lof)))

    return query_idx
