#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by cyy on 2019/10/21

import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from AL_BLVE.bt_paa import *

np.random.seed(0)

filedir ='../data/sortedUCR2018/'
class Data_Hanlder(object):
    def __init__(self, dataset_name, config):
        # read and split data
        # print('init')

        # >>>>>>>>> read txt data
        # self.data = np.loadtxt(filedir + dataset_name, delimiter=",")
        # self.label = self.data[:, 0]
        # # print(self.label)
        # self.label = np.where(self.label < 1, -1, 1)# normal:1,anomaly:-1
        # # print(self.label)
        # self.data = self.data[:, 1:]
        # self.rows, self.cols = self.data.shape

        # >>>>>>>>> read scv data
        self.data = pd.read_csv(filedir + dataset_name, sep='\t')
        df = pd.DataFrame(data=self.data)
        self.label = df[df.columns[0]].values

        # from collections import Counter
        c = Counter(self.label)
        print(c)
        # a = np.argwhere(self.label == 1).flatten()
        # b = np.argwhere(self.label == -1).flatten()
        # rate=c[self.all_labels==-1]/(c[self.all_labels==1]+c[self.all_labels==-1])
        # print(a.shape, b.shape)

        #todo：根据counter判断，将多数类转换为1，少数类转换为-1
        # self.label = np.where(self.label == 1, 1, -1) # 将原来的标签转换为1和-1
        # 统计label中的正常和异常值得标签
        c = Counter(self.label)
        b = zip(c.values(), c.keys())
        c = list(sorted(b))

        self.label = np.where(self.label == c[1][1], 1, -1)  # 将原来的标签转换为1和-1，1：normal，-1：anomaly
        c = Counter(self.label)
        print(c)

        self.data = df[df.columns[1:]].values
        self.rows, self.cols = self.data.shape
        print(dataset_name,self.data.shape)

        # plot the original series
        # plt.figure()
        # for i in range(4):
        #     if self.label[i] == -1:
        #         plt.plot(self.data[i],'r')
        #         # print(i)
        #     if self.label[i] == 1:
        #         plt.plot(self.data[i],'b')
        #         # print(i)
        # plt.show()
        # exit()

        self.time_steps = config['time_steps']  # 序列本身的长度即调用call次数
        self.pointer = 0  # todo:?
        self.train = np.array([])
        self.train_label = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
        self.win_size = config['win_size']
        self.batch_size = config['batch_size']
        self._process_source_data()
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.label, test_size=0.25,
        #                                                                         random_state=0)

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
        """变成三维[rows,time_step,cols]"""
        print('Data Arraging...')
        self.all_data = np.array([])
        self.all_labels = np.array([])
        # print('time',self.time_steps)
        # print(self.data)
        # print('label',self.label)
        self.all_data = self.data[:, np.newaxis, :]
        self.all_labels = self.label


        # indexs=0
        # for index in range(self.data.shape[0] - self.time_steps + 1):
        #
        #     this_array = self.data[index:index + self.time_steps]
        #     # print(this_array.shape)
        #     # print(self.data.shape)
        #     this_array = np.reshape(this_array, (-1, self.time_steps, self.cols))  # 变成三维
        #     # print(this_array.shape)
        #
        #     # this_array = self.data[index:index + self.time_steps].reshape((-1, self.time_steps, self.cols))
        #     # print(self.label)
        #     time_steps_label = self.label[index:index + self.time_steps]
        #     # print(time_steps_label)
        #     # exit()
        #     if np.any(time_steps_label == -1):
        #         this_label = -1
        #     else:
        #         this_label = 1
        #     # print(this_label)
        #
        #     # 将转换后的data和label组合进all_data 和all_label
        #     if self.all_data.shape[0] == 0:
        #         self.all_data = this_array
        #         self.all_labels = this_label
        #         # print('ccc',self.all_data.shape)
        #     else:
        #         self.all_data = np.concatenate([self.all_data, this_array], axis=0)# axis = 0 纵向的拼接
        #         self.all_labels = np.append(self.all_labels, this_label)
        #     indexs=index
        # self.test = self.all_data
        # self.test_label = self.all_labels

    def init_query_idx(self):
        print('init query index...')
        # labeled_idx = bt_paa_lof(self.win_size, self.data, self.batch_size)
        labeled_idx = bt_paa_lof(self.win_size, self.unlabeled_data.reshape(-1, self.cols), self.batch_size)
        print('choose label', self.unlabeled_label[labeled_idx])
        return labeled_idx

    def _split_save_data(self):
        '''
        train dataset：Unlabeled dataset、labeled dataset
        test dataset
        '''
        print('Split Data an Save ...')
        x_train, x_test, y_train, y_test = train_test_split(
            self.all_data, self.all_labels, test_size=0.7, random_state=0)
        self.unlabeled_data = x_train
        self.unlabeled_label = y_train
        print('unlabeled dataset shape:', self.unlabeled_data.shape)


        labeled_idx = self.init_query_idx()


        print(len(self.train))
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

        np.save('arrange/train.npy', self.train)
        np.save('arrange/train_label.npy', self.train_label)
        np.save('arrange/test.npy', self.test)
        np.save('arrange/test_label.npy', self.test_label)
        np.save('arrange/unlabel.npy', self.unlabeled_data)
        np.save('arrange/unlabel_label.npy', self.unlabeled_label)

    def _get_data(self):
        self._process_source_data()
        if os.path.exists('arrange/train.npy'):
            self.train = np.load('arrange/train.npy')  # 只训练正常数据
            # print(self.train.shape)
            self.train_label = np.load('arrange/train_label.npy')  # 只训练正常数据
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/test_data.npy')
            self.test_label = np.load('arrange/test_label.npy')
            # print('train data', self.train)
            # print(self.train.ndim)
            self.unlabeled_data = np.load('arrange/unlabel.npy')
            self.unlabeled_label = np.load('arrange/unlabel_label.npy')

        # 层数

        if self.train.ndim == 3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != self.cols:
                return 0

    def fetch_data(self):
        # labeled_idx=self.query_idx()

        self._split_save_data()

        # print(self.train.shape)
        return self.train
