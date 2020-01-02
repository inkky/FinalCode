# !/usr/bin/env python
# -*-coding:utf-8-*-


import os
import time
import numpy
from sklearn.metrics import roc_auc_score,mean_squared_error,coverage_error
from sklearn.preprocessing import scale, minmax_scale
from matplotlib import pyplot
from Base import *


class TSForest(object):
    def __init__(self, tree_size=2, hash_size=2):
        self.tree_size = tree_size
        self.hash_size = hash_size
        self.intervals = list()
        self.hash_familiy = list()
        self.model = None
        self.data = None

    def fit(self, data):
        rows, cols = data.shape
        t = 0
        while t < self.tree_size:
            self.intervals.append(self.init_interval(0, cols-1, depth=0, limit=numpy.log2(cols/3)))
            hash_tmp = self.init_project_coefficent(rows, cols)
            self.hash_familiy.append(hash_tmp)
            t += 1
        # deal data sets
        t = 0
        n_data = numpy.zeros((self.tree_size, rows, cols))
        while t < self.tree_size:
            for r in range(rows):
                n_data[t, r] = self.hash_val(data[r], self.hash_familiy[t][1], self.hash_familiy[t][0])
            t += 1
        self.data = n_data
        dict_data = list()
        t = 0
        self.model = list()
        while t < self.tree_size:
            one_data = list()
            for c in range(cols):
                one_col = dict()
                for r in range(rows):
                    one_col[self.data[t, r, c]] = one_col.get(self.data[t, r, c], 0) + 1
                one_data.append(one_col)
            self.model.append(one_data)
            t += 1

    def predict(self):
        scores = list()
        iters, rows, cols = self.data.shape
        for r in range(rows):
            score = 0
            score = self.__cal_score_one(r)
            scores.append(score)
        return scores

    def __cal_score_one(self, r):
        score = 0
        t = 0
        while t < self.tree_size:
            score += self.__cal_score_on_one_tree(r, t)
            t += 1
        score /= self.tree_size
        return score

    def __cal_score_on_one_tree(self, r, t):
        intervals = self.intervals[t]
        score = 0
        for lower, upper in intervals:
            s_i = 0.0
            for p in range(lower, upper+1):
                s = 0.0
                for q in range(lower, upper+1):
                    s += self.model[t][q].get(self.data[t, r, p], 0)
                    s += 0.5*self.model[t][q].get(self.data[t, r, p]-self.hash_familiy[t][0], 0)
                    s += 0.5*self.model[t][q].get(self.data[t, r, p]+self.hash_familiy[t][0], 0)
                    # s = max(s, self.model[t][q].get(self.data[t, r, p], 0))
                s /= (upper-lower+1)
                s_i += s
            # s_i /= (upper-lower+1)
            score += s_i
        # score /= len(intervals)
        return score

    def hash_val(self, x, r, w):
        res = 0
        res = numpy.floor((x+r)/w)
        # res = numpy.floor((numpy.exp(-x)+r)/w)
        return res

    def init_interval(self, start, end, depth, limit):
        if end - start <= 2 or depth >= limit:
            return [(start, end)]
        intervals = list()
        selected = numpy.random.randint(start + 1, end - 1)
        left = self.init_interval(start, selected, depth + 1, limit)
        right = self.init_interval(selected + 1, end, depth + 1, limit)
        intervals += left
        intervals += right
        return intervals

    def init_project_coefficent(self, rows, cols):
        para = numpy.random.uniform(1.0 / (rows)**0.5, 1 - 1.0 / (rows)**0.5)
        # shift = [numpy.random.uniform(0, para) for _ in range(cols)]
        shift = numpy.random.uniform(0, para)
        return (para, shift)

from collections import Counter
if __name__ == '__main__':

    dir_path = '../data/UCRtwoclass/'
    # files = ['ECG200', 'Gun_Point', 'Lighting2', 'MoteStrain',
    #          'SonyAIBORobotSurfaceII', 'ToeSegmentation1', 'ToeSegmentation2']

    for f in os.listdir(dir_path):
        f_path = dir_path + f

        # if f == 'HandOutlines' or f == 'yoga' or f == 'TwoLeadECG':
        #     continue
        # if f != 'BeetleFly.txt':
        #     continue
        print(f)

        data = numpy.loadtxt(f_path, delimiter=',')
        label = data[:, 0]
        for i in range(len(label)):
            if label[i] != 1:
                label[i] = 0
        data = data[:, 1:]
        print(Counter(label))
        rows, cols = data.shape
        data = scale(data, axis=0)
        begin = time.clock()
        forest = TSForest(tree_size=15)
        forest.fit(data)
        scores = minmax_scale(forest.predict())
        end = time.clock()
        auc = roc_auc_score(label, scores)
        score_ratio, pred = cal_score_ratio(label, scores)
        error=mean_squared_error(label, pred)
        if auc<0.5:
            auc=1-auc
        output = 'Data=%s, AUC=%f, error=%f, Time=%s' %(f, auc, error,end-begin)
        print(output)
