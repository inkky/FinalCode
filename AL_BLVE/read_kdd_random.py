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
from activelearning.modaltest.bt_paa import *
from sklearn.model_selection import KFold

np.random.seed(0)

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
        labels[labels != 'normal.'] = -1
        labels[labels == 'normal.'] = 1
        df['label'] = labels

        self.data,self.label=_to_xy(df,target='label') # 分离data和label
        self.label=self.label.flatten().astype(int)
        self.data=self.data.astype(np.float32)
        self.rows, self.cols = self.data.shape
        # print(self.data.shape)

        self.time_steps = config['time_steps']  # 序列本身的长度即调用call次数
        self.pointer = 0  # todo:?
        self.train = np.array([])
        self.train_label = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
        self.win_size = config['win_size']
        self.batch_size = config['batch_size']
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
        # print('original data shape', self.data.shape)

        # 在第二列增加一个维度
        # print('data shape:',self.data.shape)
        self.all_data = self.data[:, np.newaxis,:]
        self.all_labels=self.label
        # print('all data shape:',self.all_data.shape)
        # print('all label shape:',self.label.shape)
        #
        # self.test = self.all_data
        # self.test_label = self.all_labels

    def init_query_idx(self):
        print('init query index...')
        labeled_idx = np.random.choice(range(len(self.unlabeled_label)), size=1)
        # self.unlabeled_data=self.unlabeled_data.reshape(-1,self.cols)
        # labeled_idx = bt_paa_lof(self.win_size, self.unlabeled_data.reshape(-1,self.cols), self.batch_size)
        print('choose label', self.unlabeled_label[labeled_idx])
        return labeled_idx

    def _split_save_data(self):
        '''
        train dataset：Unlabeled dataset、labeled dataset
        test dataset
        '''
        print('Split Data an Save ...')
        x_train, x_test, y_train, y_test = train_test_split(
            self.all_data, self.all_labels, test_size=0.995,random_state=0)
        self.unlabeled_data=x_train
        self.unlabeled_label=y_train
        print('unlabeled dataset shape:',self.unlabeled_data.shape)


        labeled_idx = self.init_query_idx()

        # print(self.unlabeled_data[labeled_idx].shape)
        if len(self.train) != 0:
            self.train = np.vstack((self.train, self.unlabeled_data[labeled_idx]))
        else:
            self.train = self.unlabeled_data[labeled_idx]
        if len(self.train_label) != 0:
            self.train_label = np.hstack((self.train_label, self.unlabeled_label[labeled_idx]))
        else:
            self.train_label = self.unlabeled_label[labeled_idx]

        self.unlabeled_data = np.delete(self.unlabeled_data, labeled_idx, axis=0)
        self.unlabeled_label = np.delete(self.unlabeled_label, labeled_idx, axis=0)
        # self.data = np.delete(self.data, labeled_idx, axis=0)
        self.test=x_test
        self.test_label=y_test

        np.save('arrange/kdd_train.npy', self.train)
        np.save('arrange/kdd_train_label.npy', self.train_label)
        np.save('arrange/kdd_test.npy', self.test)
        np.save('arrange/kdd_test_label.npy', self.test_label)
        np.save('arrange/kdd_unlabel.npy', self.unlabeled_data)
        np.save('arrange/kdd_unlabel_label.npy', self.unlabeled_label)


    def _get_data(self):
        self._process_source_data()
        if os.path.exists('arrange/kdd_train.npy'):
            self.train = np.load('arrange/kdd_train.npy')  # 只训练正常数据
            # print(self.train.shape)
            self.train_label = np.load('arrange/kdd_train_label.npy')  # 只训练正常数据
            # self.train = np.load('result/train_normal.npy')# 训练正常和异常数据
            self.test = np.load('arrange/kdd_test_data.npy')
            self.test_label = np.load('arrange/kdd_test_label.npy')
            # print('train data', self.train)
            # print(self.train.ndim)
            self.unlabeled_data=np.load('arrange/kdd_unlabel.npy')
            self.unlabeled_label=np.load('arrange/kdd_unlabel_label.npy')

        # 层数
        if self.train.ndim == 3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != self.cols:
                return 0

    def fetch_data(self):
        # labeled_idx=self.query_idx()

        self._split_save_data()

        # print(self.train.shape)
        return self.train
