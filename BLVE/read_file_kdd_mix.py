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
import importlib
import scipy.io
import sys
# from data.kdd import _col_names,_encode_text_dummy,_to_xy

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().flatten().astype(int)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


# file=os.listdir(filedir)
class Data_Hanlder(object):
    def __init__(self, dataset_name, config):
        # # read data
        # data = importlib.import_module("data.{}".format('arrhythmia'))
        # Data
        col_names = _col_names()
        df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
        text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

        for name in text_l:  # 删掉这些离散的字段
            _encode_text_dummy(df, name)

        #转换label
        labels = df['label'].copy()
        labels[labels != 'normal.'] = 1
        labels[labels == 'normal.'] = -1
        df['label'] = labels

        self.data,self.label=_to_xy(df,target='label') # 分离data和label
        self.label=self.label.flatten().astype(int)
        self.data=self.data.astype(np.float32)
        self.rows, self.cols = self.data.shape
        # print(self.data.shape)

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
        print('Data Scaling MinMaxScaler...')
        # standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0, 1))
        # self.data = standscaler.fit_transform(self.data)
        self.data = mscaler.fit_transform(self.data)

    def _data_arrage(self):
        """变成三维[rows,time_step,cols]"""
        print('Data Arraging...')
        self.all_data = np.array([])
        self.all_labels = np.array([])
        # print('time',self.time_steps)
        print('original data shape', self.data.shape)

        # 在第二列增加一个维度
        print(self.data.shape)
        self.all_data = self.data[:, np.newaxis,:]
        self.all_labels=self.label
        print(self.all_data.shape)
        print(self.label.shape)


    def _split_save_data(self):
        print('Split Data an Save ...')
        from collections import Counter
        c=Counter(self.all_labels)
        print(c)
        x_train, x_test, y_train, y_test = train_test_split(self.all_data, self.all_labels, test_size=0.25,
                                                            random_state=0)


        np.save('arrange/arr_train.npy',x_train )
        np.save('arrange/arr_test.npy', x_test)
        np.save('arrange/arr_test_label.npy', y_test)

        # normal = x_train[y_train == 1]
        # abnormal = x_train[y_train == -1]
        # self.train = normal  # 只训练正常数据
        # self.train_anomaly=x_train
        # # self.train_anomaly = np.concatenate([normal, abnormal], axis=0)  # 训练正常和异常数据
        # self.test = x_test
        # self.test_label = y_test
        #
        # np.save('arrange/arr_train.npy', self.train)
        # np.save('arrange/arr_train_anomally.npy', self.train_anomaly)
        # np.save('arrange/arr_test.npy', self.test)
        # np.save('arrange/arr_test_label.npy', self.test_label)

    def _get_data(self):
        print('Get Data ...')
        self._process_source_data()
        if os.path.exists('arrange/kdd_train_normal.npy'):
            self.train = np.load('arrange/kdd_train.npy')  # 只训练正常数据
            # print(self.train.shape)
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/kdd_test_data.npy')
            self.test_label = np.load('arrange/kdd_test_label.npy')
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
