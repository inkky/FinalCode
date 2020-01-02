import os
import torch
from torch.utils.data import Dataset,TensorDataset
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from collections import Counter
import pandas as pd
from ganomaly.options import Options


def load_data_ucr(opt):
    # path = '../../../data/sortedUCR2018/'
    path =os.getcwd() + '../../data/sortedUCR2018/'
    dataset_name='ECGFiveDays.tsv'
    data=pd.read_csv(path + dataset_name, sep='\t')
    df = pd.DataFrame(data=data)
    data_label = df[df.columns[0]].values
    c = Counter(data_label)
    print(c)
    b = zip(c.values(), c.keys())
    c = list(sorted(b))
    data_label = np.where(data_label == c[1][1], 1, 0)  # 将原来的标签转换为1和-1，1：normal，0：anomaly

    dataset = df[df.columns[1:]].values
    rows, cols = data.shape


    # dataset = scipy.io.loadmat(path)
    # data_label = dataset['y']  # (452, 1)
    # dataset = dataset["X"]  # (452, 274)
    # # 本身normal：0，anomaly：1 ---》 normal:1,anomaly:-1 ；满足条件(condition)，输出x，不满足输出y。
    # # data_label = np.where(data_label == 0, 1, -1)
    # data_label = data_label.flatten().astype(int)
    # # data_label = pd.Series(data_label) # 1为正常样本，-1为异常样本
    # # print(data_label.value_counts(normalize=False, dropna=False))

    print('Data Scaling MinMaxScaler...')
    # standscaler = StandardScaler()
    mscaler = MinMaxScaler(feature_range=(0, 1))
    # self.data = standscaler.fit_transform(self.data)
    dataset = mscaler.fit_transform(dataset)

    x_train, x_test, y_train, y_test = train_test_split(dataset, data_label, test_size=0.4,
                                                        random_state=0)
    ucr_normal_train_data = x_train[y_train == 0]
    ucr_abnormal_train_data = x_train[y_train == 1]
    ucr_test_data = x_test
    ucr_test_label = y_test
    ucr_normal_train_label = np.zeros((ucr_normal_train_data.shape[0], ), dtype='int32')

    ucr_normal_train_data = ucr_normal_train_data[:, np.newaxis, :]
    ucr_test_data = ucr_test_data[:, np.newaxis, :]
    print(ucr_normal_train_data.shape)
    print(ucr_test_data.shape)

    train_dataset = TensorDataset(torch.from_numpy(ucr_normal_train_data),
                                  torch.from_numpy(ucr_normal_train_label))
    test_dataset = TensorDataset(torch.from_numpy(ucr_test_data),
                                 torch.from_numpy(ucr_test_label))

    dataset = {}
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


if __name__ == '__main__':
    opt = Options().parse()
    load_data_ucr(opt)