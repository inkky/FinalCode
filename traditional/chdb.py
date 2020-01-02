# !/usr/bin/env python
# -*-coding:utf-8-*-


import numpy as np
from matplotlib import pyplot as plt
from EITable import *
from sklearn.preprocessing import minmax_scale
from TSLSH import TSForest


def read_data(name, sig, sub_len):
    data = np.loadtxt('../codes/mitbih/data/'+name)
    data = data[:, sig]
    length = len(data)
    if length > 3750:
        data = data[:3750]
    ndata = data.reshape(int(length/sub_len), sub_len)
    if name == 'chfdb_chf01_275.txt' and sig == 2:
        ndata = scale(ndata, axis=1)
    return data, ndata


def cal_score_by_table(data, k):
    rows, cols = data.shape
    t_data = preprocessing(data, k)
    predict = cal_score(data, t_data, cols, k)
    scores = list()
    for x in predict:
        tmp = [x]*cols
        scores += tmp
    scores = minmax_scale(scores)
    return scores


def cal_score_by_tslsh(data):
    rows, cols = data.shape
    # data = scale(data, axis=0)
    forest = TSForest(tree_size=25)
    forest.fit(data)
    predict = forest.predict()
    scores = list()
    for x in predict:
        tmp = [x] * cols
        scores += tmp
    scores = minmax_scale(scores)
    return scores


if __name__ == '__main__':

    name = 'chfdb_chf01_275.txt'
    # name = 'chfdb_chf13_45590.txt'
    signal = 2
    abnormal_mark = {'chfdb_chf01_275.txt':9, 'chfdb_chf13_45590.txt':11}
    r_data, data = read_data(name, sig=signal, sub_len=250)
    rows, cols = data.shape
    predict_by_table = cal_score_by_table(data, k=5)
    predict__by_tlsh = cal_score_by_tslsh(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    plt.tight_layout()
    # plt.yticks([])
    plt.subplot(211)
    plt.plot(range(rows*cols), r_data)
    a_start = abnormal_mark.get(name)
    plt.plot(range(cols*a_start, cols*(a_start+1)), r_data[cols*a_start:cols*(a_start+1)], color='R', linewidth=4)
    plt.title('(a)', loc='right')

    plt.subplot(212)
    plt.plot(range(rows*cols), [1-x for x in predict_by_table], color='#9e026d')
    plt.annotate('EITable', xy=(0.8,0.5))
    plt.title('(b)', loc='right')

    # plt.subplot(313)
    # plt.plot(range(rows * cols), [1 - x for x in predict__by_tlsh], color='#cc2f69')
    # plt.annotate('DLDE', xy=(0.8, 0.5))
    # plt.title('(c)', loc='right')
    # plt.show()
    plt.savefig('../results/figures/'+name+'_'+str(signal)+'EITable.png', dpi=300)
