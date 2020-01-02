import os
import torch
from torch.utils.data import Dataset,TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from collections import Counter

from ganomaly.options import Options


def load_data_kdd(opt):

    dataset = {}

    col_names = _col_names()
    path = os.path.join('data/KDD99/kddcup.data_10_percent_corrected', )
    df = pd.read_csv(path, header=None, names=col_names, engine='python')
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:  # 将离散字段转换为onehot形式并删掉离散的字段
        _encode_text_dummy(df, name)

    print(df['label'].value_counts(normalize=False, dropna=False))
    # df = df[~df['label'].isin(['smurf.'])]
    # df = df[~df['label'].isin(['neptune.'])]
    # print(df['label'].value_counts(normalize=False, dropna=False))
    # 除去两种数量较多的DDOS攻击类型后，有97278个正常样本，8752个异常样本


    # 转换label
    labels = df['label'].copy()

    labels[labels != 'normal.'] = 1  # 1为异常
    labels[labels == 'normal.'] = 0  # 0为正常
    df['label'] = labels


    # 分离data和label
    kdd_label = df['label'].astype(int)
    kdd_data = df.drop('label', axis=1).astype(np.float32)
    print(kdd_data.shape)
    kdd_label = kdd_label.values
    kdd_data = kdd_data.values

    print('Data Scaling MinMaxScaler...')
    # standscaler = StandardScaler()
    mscaler = MinMaxScaler(feature_range=(0, 1))
    # self.data = standscaler.fit_transform(self.data)
    kdd_data = mscaler.fit_transform(kdd_data)

    x_train, x_test, y_train, y_test = train_test_split(kdd_data, kdd_label, test_size=0.6,
                                                        random_state=0)
    kdd_normal_train_data = x_train[y_train == 1]
    kdd_abnormal_train_data = x_train[y_train == 0]  # 训练数据不需要label
    kdd_normal_abnormal_test_data = x_test
    for x in range(len(y_test)):
        if y_test[x] == 1:
            y_test[x] = 0
        elif y_test[x] == 0:
            y_test[x] = 1
        else:
            print('label error!')
    kdd_normal_abnormal_test_label = y_test


    # print(kdd_normal_train_data.shape)
    # print(kdd_abnormal_train_data.shape)
    # print(kdd_normal_abnormal_test_data.shape)
    # print(kdd_normal_abnormal_test_label.shape)

    # 0为正常样本 1为异常样本
    kdd_normal_train_label = np.zeros((kdd_normal_train_data.shape[0], ), dtype='int32')

    kdd_normal_train_data = kdd_normal_train_data[:, np.newaxis, :]
    kdd_normal_abnormal_test_data = kdd_normal_abnormal_test_data[:, np.newaxis, :]


    train_dataset = TensorDataset(torch.from_numpy(kdd_normal_train_data),
                                  torch.from_numpy(kdd_normal_train_label))
    test_dataset = TensorDataset(torch.from_numpy(kdd_normal_abnormal_test_data),
                                 torch.from_numpy(kdd_normal_abnormal_test_label))
    dataset['train'] = train_dataset
    dataset['test'] = test_dataset

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}

    # print('data shape testing=========={}'.format(Variable(dataset['train']).data.size()))
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x]) for x in splits}
    print('finished!')
    return dataloader


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


if __name__ == '__main__':
    opt = Options().parse()
    load_data_kdd(opt)