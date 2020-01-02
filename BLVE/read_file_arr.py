# -*- coding: utf-8 -*-
# @Author: cyy
# @Date  : 2019/3/15

'''
Arrhythmia
This database contains 279 attributes, 206 of which are linear valued and the rest are nominal.
addredd: http://archive.ics.uci.edu/ml/datasets/arrhythmia
Number of Instances: 452
Number of Attributes: 279

Class code :          Class   :                  Number of instances:
       01             Normal				                      245
       02             Ischemic changes (Coronary Artery Disease)   44
       03             Old Anterior Myocardial Infarction           15
       04             Old Inferior Myocardial Infarction           15
       05             Sinus tachycardy			                   13
       06             Sinus bradycardy			                   25
       07             Ventricular Premature Contraction (PVC)       3
       08             Supraventricular Premature Contraction	    2
       09             Left bundle branch block 		                9
       10             Right bundle branch block		               50
       11             1. degree AtrioVentricular block	            0
       12             2. degree AV block		                    0
       13             3. degree AV block		                    0
       14             Left ventricule hypertrophy 	                4
       15             Atrial Fibrillation or Flutter	            5
       16             Others				                       22


In this experiment:
For the arrhythmia dataset, anomalous classes
represent 15% of the data and therefore the 15% of samples
with the highest anomaly scores are likewise classified as
anomalies (positive class). v

In this paper:
然而所有数据集已经经过处理变成0,1，0为正常386，1为异常66
做不出paper的结果，太奇怪了
'''

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

# file=os.listdir(filedir)
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

        ##########=============================###############################

        # self.train, self.train_label = data.get_train()  # 这个label并没有被用到？？？y！！！
        # train_copy = self.train.copy()
        # self.test, self.test_label = data.get_test()
        # print('trainx.shape:', self.train.shape)
        # print(self.train)  # [ 40.    1.  153.  ...   2.5  35.3  57.3] 很奇怪，不需要归一化？
        # self.rows, self.cols = self.train.shape
        # print(train_label)


        self.time_steps = config['time_steps']  # 序列本身的长度即调用call次数
        self.pointer = 0  # todo:?
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
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
        self.all_data = np.array([])
        self.all_labels = np.array([])
        # print('time',self.time_steps)
        print('data shape', self.data.shape)
        # print(self.data)
        # print('label',self.label)

        for index in range(self.data.shape[0] - self.time_steps + 1):

            this_array = self.data[index:index + self.time_steps]
            this_array = np.reshape(this_array, (-1, self.time_steps, self.cols))  # 变成三维
            # print(this_array)
            # print(self.label)
            # this_array = self.data[index:index + self.time_steps].reshape((-1, self.time_steps, self.cols))
            time_steps_label = self.label[index:index + self.time_steps]
            # print(time_steps_label)


            #
            if np.any(time_steps_label == -1):
                this_label = -1
            else:
                this_label = 1
            # print(this_label)

            # 将转换后的data和label组合进all_data 和all_label
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.all_labels = this_label
            else:
                self.all_data = np.concatenate([self.all_data, this_array], axis=0)
                self.all_labels = np.append(self.all_labels, this_label)
        # print('all_data',self.all_data.shape)
        # print('all label',self.all_labels)


    def _split_save_data(self):
        print('split data')
        x_train, x_test, y_train, y_test = train_test_split(self.all_data, self.all_labels, test_size=0.25,
                                                            random_state=0)

        #train normal data
        normal = x_train[y_train == 1]
        abnormal = x_train[y_train == -1]
        self.train = normal  # 只训练正常数据
        self.train_anomaly = np.concatenate([normal, abnormal], axis=0)  # 训练正常和异常数据
        self.test = x_test
        self.test_label = y_test
        print(Counter(self.test_label))

        np.save('arrange/arr_train.npy', self.train)
        np.save('arrange/arr_train_anomally.npy', self.train_anomaly)
        np.save('arrange/arr_test_data.npy', self.test)
        np.save('arrange/arr_test_label.npy', self.test_label)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



        # train mixed data
        # np.save('arrange/arr_train.npy', x_train)
        # np.save('arrange/arr_train_anomally.npy', y_train)
        # np.save('arrange/arr_test_data.npy', x_test)
        # np.save('arrange/arr_test_label.npy', y_test)

    def _get_data(self):
        self._process_source_data()
        if os.path.exists('arrange/arr_train.npy'):
            self.train = np.load('arrange/arr_train.npy')  # 只训练正常数据
            # print(self.train.shape)
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/arr_test_data.npy')
            self.test_label = np.load('arrange/arr_test_label.npy')
            # print('train data', self.train)
            # print(self.train.ndim)

        # 层数

        if self.train.ndim == 3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != self.cols:
                return 0

    def fetch_data(self, batch_size):
        # print('tr', self.train.shape)
        if self.train.shape[0] == 0:
            self._get_data()
            # print('train', self.train.shape)

        if self.train.shape[0] < batch_size:
            return_train = self.train
        else:
            if (self.pointer + 1) * batch_size >= self.train.shape[0] - 1:
                self.pointer = 0
                return_train = self.train[self.pointer * batch_size:, ]
            else:
                self.pointer = self.pointer + 1
                return_train = self.train[self.pointer * batch_size:(self.pointer + 1) * batch_size, ]
        if return_train.ndim < self.train.ndim:
            return_train = np.expand_dims(return_train, 0)

        return return_train

    def plot_confusion_matrix(self, y_true, y_pred, labels, title):
        cmap = plt.cm.binary
        cm = confusion_matrix(y_true, y_pred)
        tick_marks = np.array(range(len(labels))) + 0.5
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 4), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        intFlag = 0
        for x_val, y_val in zip(x.flatten(), y.flatten()):

            if (intFlag):
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=10, va='center', ha='center')

            else:
                c = cm_normalized[y_val][x_val]
                if (c > 0.01):
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
        if (intFlag):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels)
        plt.yticks(xlocations, labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        plt.show()

    def plot_figure(self, ori_data, rec_data):
        """画图，真实的图和重构的图"""
        plt.figure()
        ori_data = np.reshape(ori_data, [-1, self.cols])
        rec_data = np.reshape(rec_data, [-1, self.cols])
        print('ori_data.shape', ori_data.shape)
        print('rec_data.shape', rec_data.shape)
        plt.plot(ori_data[1])
        plt.plot(rec_data[1])
        plt.show()

