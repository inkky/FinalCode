# -*- coding: utf-8 -*-
# @Author: cyy
# @Date  : 2019/3/15

import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# 二分类
# filedir = "F:\\GraduationProject/data/UCRtwoclass/"
# filedir = os.getcwd() + '/sortedUCR2018/'
filedir = os.getcwd() + '/../data/sortedUCR2018/'


# def readfile():
#     for filename in os.listdir(filedir):
#         data = np.loadtxt(filedir + filename, delimiter=",")
#         print(data)
#         labels = data[:, 0]
#         labels = np.where(labels < 1, -1, 1)  # normal:1,anomaly:-1
#         data = data[:, 1:]
#         rows, cols = data.shape
#         print(labels.shape)
#         # 随机选择25%作为测试集，剩余作为训练集
#         x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
#         print(y_train, y_test)
#         return data, filename


# file=os.listdir(filedir)
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
        # c = Counter(self.label)
        # print(c)
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


        self.data = df[df.columns[1:]].values
        self.rows, self.cols = self.data.shape
        print(dataset_name,self.data.shape)

        self.time_steps = config['time_steps']  # 序列本身的长度即调用call次数
        self.pointer = 0  # todo:?
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
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

        self.all_data = np.array([])
        self.all_labels = np.array([])
        # print('time',self.time_steps)
        # print(self.data)
        # print('label',self.label)


        indexs=0
        for index in range(self.data.shape[0] - self.time_steps + 1):

            this_array = self.data[index:index + self.time_steps]
            # print(this_array.shape)
            # print(self.data.shape)
            this_array = np.reshape(this_array, (-1, self.time_steps, self.cols))  # 变成三维
            # print(this_array.shape)

            # this_array = self.data[index:index + self.time_steps].reshape((-1, self.time_steps, self.cols))
            # print(self.label)
            time_steps_label = self.label[index:index + self.time_steps]
            # print(time_steps_label)
            # exit()
            if np.any(time_steps_label == -1):
                this_label = -1
            else:
                this_label = 1
            # print(this_label)

            # 将转换后的data和label组合进all_data 和all_label
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.all_labels = this_label
                # print('ccc',self.all_data.shape)
            else:
                self.all_data = np.concatenate([self.all_data, this_array], axis=0)# axis = 0 纵向的拼接
                self.all_labels = np.append(self.all_labels, this_label)
            indexs=index



    def _split_save_data(self):
        # print('split data')
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.all_data, self.all_labels, test_size=0.2,
        #                                                                                 random_state=0)
        # # 只训练正常的数据
        # self.train = self.x_train[self.y_train == 1]
        # self.abnormal = self.x_train[self.y_train == -1]
        #
        # print('train', self.x_train.shape)
        # print('normal', self.train.shape)
        # print('abnormal', self.abnormal.shape)
        # print('test', self.x_test.shape)

        # >>>>>>>>>>>>>>>todo：因为异常的比例过高，需要调整

        x_train, x_test, y_train, y_test = train_test_split(self.all_data, self.all_labels, test_size=0.75, #0.15
                                                            random_state=0)

        # normal = x_train[y_train == 1]
        # abnormal = x_train[y_train == -1]
        #
        # self.train = normal  # 只训练正常数据
        # self.train_anomaly = np.concatenate([normal, abnormal], axis=0)  # 训练正常和异常数据
        # self.test = x_test
        # self.test_label = y_test
        # print(Counter(self.test_label))

        # normal = self.all_data[self.all_labels == 1]
        # abnormal = self.all_data[self.all_labels == -1]
        # print(normal.shape)
        # 将测试机为异常数量的两倍
        # split_no = normal.shape[0] - abnormal.shape[0]
        # split_no = int(split_no / 3)
        # print("split_no:", split_no)
        # print("normal:", self.train.shape)
        # print("abnormal:", self.train_anomaly.shape)
        # print("all-data:", self.all_data.shape)
        # print('test data', self.test.shape)
        # print('test label', self.test_label.shape)

        # 只训练正常数据
        # self.train = normal[:split_no, :]
        # print('1', self.train.shape)

        # 训练正常和异常数据
        # self.train_anomaly = np.concatenate([normal[split_no:, :], abnormal], axis=0)

        # self.test = np.concatenate([normal[split_no:, :], abnormal], axis=0)
        # self.test_label = np.concatenate([np.ones(normal[split_no:, :].shape[0]), -np.ones(abnormal.shape[0])])

        # np.save('arrange/train.npy', self.train)
        # np.save('arrange/train_anomally.npy', self.train_anomaly)
        # np.save('arrange/test_data.npy', self.test)
        # np.save('arrange/test_label.npy', self.test_label)

        #保存混合数据
        np.save('arrange/train.npy', x_train)
        np.save('arrange/test_data.npy', x_test)
        np.save('arrange/test_label.npy', y_test)


    def _get_data(self):
        self._process_source_data()
        if os.path.exists('arrange/train.npy'):
            self.train = np.load('arrange/train.npy')  # 只训练正常数据
            # print(self.train.shape)
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/test_data.npy')
            self.test_label = np.load('arrange/test_label.npy')
            # print('train data', self.train)


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
