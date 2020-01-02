# !/usr/bin/env python
# -*-coding:utf-8-*-
"""
PAPR
A Piecewise Aggregate pattern representation approach for anomaly detection in time series
异常子序列？
1. PAA分段
2. 每一段进行高斯分布的区域划分vertically（sax）
3. number, the mean and the variance of points falling within each region
4. matrix[区域中的点个数，方差，均值]，行为区域个数
5. !相似矩阵S，归一化：S=wd*Sd+wc*Sc+wr*sr,wd+wc+wr=1
6. random walk:如果两个点相似，他们会有边；边越少，该点越可能是异常点
    ref：Outlier Detection Using Random Walks
"""

import os
import numpy
import time
from Base import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error,roc_curve
from collections import Counter

splits = dict()
splits[2] = [0, float('inf')]
splits[3] = [-0.43, 0.43, float('inf')]
splits[4] = [-0.67, 0, 0.67, float('inf')]
splits[5] = [-0.84, -0.25, 0.25, 0.84, float('inf')]
splits[6] = [-0.97, -0.43, 0, 0.43, 0.97, float('inf')]
splits[7] = [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, float('inf')]
splits[8] = [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, float('inf')]
splits[9] = [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, float('inf')]
splits[10] = [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28, float('inf')]


def cal_matrix(data, k):
    # print(data)
    points = splits[k]
    # print(points)
    new_data = list()
    for item in data:
        tmp_points = list()
        for i in range(k):
            tmp_points.append(list())
        for p in item:
            for w in range(k):
                if p < points[w]:
                    tmp_points[w].append(p)
                    break
        tmp_matrix = numpy.zeros((k, 3))
        for w in range(k):
            tmp_matrix[w, 0] = len(tmp_points[w])
            if tmp_matrix[w, 0] != 0:
                tmp_matrix[w, 1] = numpy.mean(tmp_points[w])
                tmp_matrix[w, 2] = numpy.var(tmp_points[w])
        new_data.append(tmp_matrix)

    return numpy.array(new_data)


def cal_similarity(matrix, length, wd, wc, wr, widths):
    index = range(length)
    sim_matrix = numpy.zeros((length, length))
    for r in index:
        for c in index:
            sd = cal_d_sim(matrix[r, :, 0], matrix[c, :, 0])
            sc = cal_rc_sim(matrix[r, :, 1], matrix[c, :, 1], widths[r])
            sr = cal_rc_sim(matrix[r, :, 2], matrix[c, :, 2], widths[r])
            sim_matrix[r, c] = wd*sd + wc*sc + wr*sr
    return sim_matrix


def cal_d_sim(one, two):
    m = numpy.sum(one)
    length = len(one)
    s = 0
    for l in range(length):
        s += min(one[l], two[l])
    return 1.0*s/m


def cal_rc_sim(one, two, w=0.005):
    return numpy.exp(-1.0*numpy.linalg.norm(one-two, ord=2) / numpy.power(w, 2))


def random_walk(sim_matrix, label, error=0.1):
    """
    Outlier Detection Using Random Walks
    Algorithm OutRank-a
    """
    rows, cols = sim_matrix.shape
    s_matrix = numpy.zeros((rows, cols))
    for i in range(rows):
        totSim = 0.0
        for j in range(cols):
            totSim += sim_matrix[i, j]
        for j in range(cols):
            s_matrix[i, j] = 1.0*sim_matrix[i, j] / totSim

    damping_factor = 0.1
    ct = numpy.array([1.0/rows]*rows) # arbitrary assignment
    recursive_err = error+1
    times = 0
    while recursive_err > error and times < 100:
        ct1 = damping_factor/rows + numpy.dot(s_matrix.T, ct)*(1-damping_factor)
        recursive_err = numpy.linalg.norm(ct-ct1, ord=1)
        times += 1
        ct = ct1[:]
    return ct


def find_best_w(data):
    rows, cols = data.shape
    alist, blist = numpy.zeros(rows), numpy.zeros(rows)
    r_index = range(rows)
    gama = (5**0.5-1)/2
    coe = (2**0.5)/3
    for i in r_index:
        min_dist, max_dist = float('inf'), -float('inf')
        for j in r_index:
            if i == j:
                continue
            dist = numpy.linalg.norm(data[i]-data[j], ord=2)
            min_dist = min(dist, min_dist)
            max_dist = max(dist, max_dist)
        alist[i], blist[i] = coe*min_dist, coe*max_dist
    left, right = cal_sig(alist, blist, gama)
    ent_left = cal_entropy(left)
    ent_right = cal_entropy(right)
    epison = 1
    times = 0
    while numpy.linalg.norm(alist-blist) < 1 and times < 20:
        if ent_left < ent_right:
            blist, right = right.copy(), left.copy()
            ent_right = ent_left
            left = alist + (1-gama)*(blist-alist)
            ent_left = cal_entropy(left)
        else:
            alist, left = left.copy(), right.copy()
            ent_left = ent_right
            right = alist + gama*(blist-alist)
            ent_right = cal_entropy(right)

        times += 1

    if ent_left < ent_right:
        return left
    else:
        return right


def cal_sig(alist, blist, gama):
    length = len(alist)
    index = range(length)
    left, right = numpy.zeros(length), numpy.zeros(length)
    for i in index:
        left[i] = alist[i] + (1-gama)*(blist[i]-alist[i])
        right[i] = alist[i] + gama*(blist[i]-alist[i])
    return left, right


def cal_entropy(list):
    total = sum(list)
    list /= total
    log_list = numpy.log(list)
    return -numpy.dot(list, log_list)

def result_write(name,label,pred):
    with open('../traditionalresult/PAPR_result.txt','a') as result:
        result.write('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
                  (name,precision_score(label, pred, average='macro'),
                   recall_score(label, pred, average='macro'),
                   f1_score(label, pred, average='macro'),
                   roc_auc_score(label, pred, ),
                   mean_squared_error(label, pred)
                   ))
        result.write('\n')

if __name__ == '__main__':
    # data_sets = ['ECG200', 'Gun_Point', 'HandOutlines', 'Lighting2', 'MoteStrain', 'ShapeletSim',
    #              'SonyAIBORobotSurfaceII', 'Strawberry',
    #              'ToeSegmentation1', 'TwoLeadECG', 'optdigits', 'SyntheticControl']
    # data_sets=['ECG200']

    dir_path = '../data/UCRtwoclass/'
    # config = read_config('../config/PAPR')
    for f in os.listdir(dir_path):
        file = dir_path + f
        print(f)
        # if f != data_sets:
        #     continue
        data = numpy.loadtxt(file, delimiter=',')
        label = data[:, 0]
        data = data[:, 1:]
        rows, cols = data.shape
        c = Counter(label)
        print(c)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))
        label = numpy.where(label == c[1][1], 1, 0)
        # for i in range(len(label)):
        #     if label[i] != 1:
        #         label[i] = 0
        data = scale(data, axis=1)

        # k = config.get(f)
        k=8 #k 高斯分布划分区域
        # widths = find_best_w(data)
        # start = time.clock()
        # matrix = cal_matrix(data, k)
        # sim_matrix = cal_similarity(matrix=matrix, wc=0.3, wd=0.4, wr=0.3, length=rows, widths=widths)
        # scores = random_walk(sim_matrix, label, error=0.05)
        # end = time.clock()
        # score_ratio = cal_score_ratio(label, scores)
        # auc = roc_auc_score(label, scores)
        # fpr, tpr, thresholds = roc_curve(label, scores, pos_label=2)



        try:
            widths = find_best_w(data)
            start = time.clock()
            matrix = cal_matrix(data, k)
            sim_matrix = cal_similarity(matrix=matrix, wc=0.3, wd=0.4, wr=0.3, length=rows, widths=widths)
            scores = random_walk(sim_matrix, label, error=0.05)

            end = time.clock()
            score_ratio,pred = cal_score_ratio(label, scores)
            auc = roc_auc_score(label, scores)
            print("Data=%s, AUC=%f, Score_ratio=%f, Time=%f" % (f, auc, score_ratio, (end - start)))

            print('name: %s, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
                  (f,precision_score(label, pred, average='macro'),
                   recall_score(label, pred, average='macro'),
                   f1_score(label, pred, average='macro'),
                   roc_auc_score(label, pred, ),
                   mean_squared_error(label, pred)
                   ))
            result_write(f,label,pred)

        except Exception as e:
            output = 'Data=%s, AUC=%s, Score_ratio=%f, Time=%s' %(f, 'NAN', 'NAN', 'inf')

        # print("Data=%s, AUC=%f, Score_ratio=%f, Time=%f" % (f, auc, score_ratio, (end-start)))