#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by cyy on 2019/10/20

import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import importlib
import scipy.io
from collections import Counter
from activelearning.modaltest.bt_paa import *


np.random.seed(42)


class Data_Hanlder(object):
    def __init__(self, dataset_name, config):
        # # read data
        # data = importlib.import_module("data.{}".format('arrhythmia'))
        # Data
        self.data = scipy.io.loadmat("data/arrhythmia.mat")
        self.label= self.data['y']  # (452, 1)
        self.data = self.data["X"]  # (452, 274)
        # 本身normal：0，anomaly：1 ---》 normal:1,anomaly:-1 ；满足条件(condition)，输出x，不满足输出y。
        self.label = np.where(self.label == 0, 1, -1)

        self.label=self.label.flatten().astype(int)
        self.rows, self.cols = self.data.shape

        c = Counter(self.label)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))
        c = Counter(self.label)
        print(c)

        # plot the original series
        # plt.figure()
        # for i in range(10):
        #     if self.label[i] == -1:
        #         plt.plot(self.data[i], 'r')
        #         # print(i)
        #     if self.label[i] == 1:
        #         plt.plot(self.data[i], 'b')
        #         # print(i)
        # plt.show()
        # exit()

        self.time_steps = config['time_steps']  # 序列本身的长度即调用call次数
        self.pointer = 0  # todo:?
        self.train = np.array([])  # labeled data
        # print(self.train)
        self.train_label = np.array([])
        self.test = np.array([])  # unlabeled data
        self.test_label = np.array([])
        # self.all_data = np.array([])
        # self.all_labels = np.array([])
        self.win_size=config['win_size']
        self.batch_size = config['batch_size']
        self._process_source_data()

    def _process_source_data(self):

        self._data_scale()
        self._data_arrage()
        self._split_save_data()

    def _data_scale(self):
        """归一化"""
        # print('data_scale')
        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0, 1))
        self.data = standscaler.fit_transform(self.data)
        self.data = mscaler.fit_transform(self.data)

    def _data_arrage(self):
        """变成三维[rows,1,cols]"""
        print('Data Arraging...')
        self.all_data = np.array([])
        self.all_labels = np.array([])
        # print('time',self.time_steps)
        # print('data shape', self.data.shape)
        # print(self.data)
        # print('label',self.label)
        self.all_data = self.data[:, np.newaxis, :]
        self.all_labels = self.label


    def init_query_idx(self):
        print('init query index...')
        # labeled_idx = bt_paa_lof(self.win_size, self.data, self.batch_size)
        # 随机抽取5个点
        labeled_idx = np.random.choice(range(len(self.unlabeled_label)), size=1)
        # X_training, y_training = X[initial_idx], y[initial_idx]

        # labeled_idx = bt_paa_lof(self.win_size, self.unlabeled_data.reshape(-1, self.cols), self.batch_size)
        print('choose label', self.unlabeled_label[labeled_idx])
        return labeled_idx

    def _split_save_data(self):
        # win_size = 4
        # batch_size = 5
        # labeled_idx = bt_paa_lof(win_size, self.data, batch_size)
        print('Split Data an Save ...')
        x_train, x_test, y_train, y_test = train_test_split(
            self.all_data, self.all_labels, test_size=0.4, random_state=0)
        self.unlabeled_data = x_train
        self.unlabeled_label = y_train
        print('unlabeled dataset shape:', self.unlabeled_data.shape)


        labeled_idx = self.init_query_idx()


        # print(len(self.train))

        if len(self.train)!=0:
            self.train = np.vstack((self.train, self.unlabeled_data[labeled_idx]))
        else:
            self.train = self.unlabeled_data[labeled_idx]
        if len(self.train_label)!=0:
            self.train_label = np.hstack((self.train_label, self.unlabeled_label[labeled_idx]))
        else:
            self.train_label=self.unlabeled_label[labeled_idx]

        self.unlabeled_data = np.delete(self.unlabeled_data, labeled_idx, axis=0)
        self.unlabeled_label = np.delete(self.unlabeled_label, labeled_idx, axis=0)
        # self.data=np.delete(self.data,labeled_idx, axis=0)
        self.test = x_test
        self.test_label = y_test

        np.save('arrange/arr_train.npy', self.train)
        np.save('arrange/arr_train_label.npy', self.train_label)
        np.save('arrange/arr_test.npy', self.test)
        np.save('arrange/arr_test_label.npy', self.test_label)
        np.save('arrange/arr_unlabel.npy', self.unlabeled_data)
        np.save('arrange/arr_unlabel_label.npy', self.unlabeled_label)

    def _get_data(self):
        self._process_source_data()
        if os.path.exists('arrange/arr_train.npy'):
            self.train = np.load('arrange/arr_train.npy')  # 只训练正常数据
            # print(self.train.shape)
            self.train_label = np.load('arrange/arr_train_label.npy')  # 只训练正常数据
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/arr_test_data.npy')
            self.test_label = np.load('arrange/arr_test_label.npy')
            # print('train data', self.train)
            # print(self.train.ndim)
            self.unlabeled_data = np.load('arrange/arr_unlabel.npy')
            self.unlabeled_label = np.load('arrange/arr_unlabel_label.npy')

        # 层数

        if self.train.ndim == 3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != self.cols:
                return 0

    def fetch_data(self):
        # labeled_idx=self.query_idx()

        self._split_save_data()

        # print(self.train.shape)
        return self.train

