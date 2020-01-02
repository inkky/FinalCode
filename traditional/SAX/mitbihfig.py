# -*- coding: utf-8 -*-
'''
draw mitdbx108
'''
import time

import matplotlib.pyplot as plt
import numpy as np

from SAX.sax_knn import zscore,PAA,paa_inv,paa2letter,compareTS
from sklearn.preprocessing import minmax_scale

def sax(data, win_size, alphabetSize):

    # zscore
    data_norm = zscore(data)
    # plt.plot(data_norm[1], '-', label='zscore data')

    # paa
    paa = list()
    paa_trans = list()
    paa_alpha = list()
    bit_data = list()
    trainBeginTime = time.clock()
    ts = list()  # SAX_TD 存放starting point 和mean的差值
    te = list()  # SAX_TD 存放ending point 和mean的差值

    for d in data_norm:
        data_paa = PAA(d, win_size)
        data_paa_inv = paa_inv(data_paa, win_size)

        # print(data_paa)
        # print(len(d)/win_size)

        ########## SAX_TD ###################
        # save the begin and end time point value

        i = 0
        tmpts = list()
        tmpte = list()
        while i < len(d) / win_size:
            tmpts.append(d[i * win_size] - data_paa[i])
            tmpte.append(d[(i + 1) * win_size - 1] - data_paa[i])
            i = i + 1



        ############# sax trend ##############
        # save the relative binary trend

        bit_tmp = ''
        paa_tmp = list()

        for i in range(len(d)):
            if d[i] < data_paa_inv[i]:
                bit_tmp += '0'
            else:
                bit_tmp += '1'
        # print(bit_tmxp)
        # print(len(bit_tmp))
        bit_data.append(int(bit_tmp, 2))

        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)
        ts.append(tmpts)
        te.append(tmpte)

    paa_trans = np.array(paa_trans)
    paa = np.array(paa)
    paa_alpha = np.array(paa_alpha)
    ts = np.array(ts)
    te = np.array(te)

    # compare the two series
    raw_dist = np.zeros((rows, rows))
    # print(paa_alpha)
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # print(paa_alpha[i])
            # print(paa_alpha[j])

            ######## raw sax distance #########
            raw_dist[i, j] = compareTS(paa_alpha[i], paa_alpha[j], alphabetSize)
            # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            ########### calculate bit distance ##########
            c = bit_data[i] ^ bit_data[j]
            ones = 0
            while c:
                ones += 1
                c &= (c - 1)

            raw_dist[i, j] += np.sqrt(ones * 1.0 / win_size)  # bit dist

            ##### sax_td distance #####
            # raw_dist[i,j]+=np.sqrt(np.sum(np.square(ts[i]-ts[j])+np.square(te[i]-te[j]))) #sax_td dist

            raw_dist[j, i] = raw_dist[i, j]

    return raw_dist



class StringsAreDifferentLength(Exception): pass





d_file = '../data/mitbih/mitdbx108.txt'
obj = open(d_file, 'r')
obj.readline()
obj.readline()
data = list()
times = list()
for line in obj.readlines():
    items = [x.strip() for x in line[:-1].split('\t')]
    times.append(items[0])
    data.append([float(items[1]), float(items[2])])
obj.close()

d_file = '../data/mitbih/mitdbx108_annotations.txt'
obj = open(d_file, 'r')
obj.readline()
annotations = dict()
for line in obj.readlines():
    item = line.split(' ')
    items = list()
    for x in item:
        if x.strip() != '':
            items.append(x.strip())
    annotations[items[0]] = items[2]

length = 5400

ann_x, ann_y = list(), list()
for key in annotations.keys():
    for i in range(length):
        if times[i] == key:
            ann_x.append(i)
            ann_y.append(annotations.get(key))
            break

sub_len = 300
data = data[:length]
data = np.array(data)
data = data[:, 0]
ndata = data.reshape((int(length/sub_len), sub_len))

rows, cols = ndata.shape
print(ndata.shape)

# win_size=10,15
win_size = 6
alphabetSize = 4
rawdist = sax(ndata, win_size, alphabetSize)
score=list()
for i in range(rows):
    print(np.mean(rawdist[i]))
    score.append(np.mean(rawdist[i]))
score=minmax_scale(score)
print(score)


plt.figure()

plt.subplot(211)
plt.plot(range(length), data)
a_points = list()
for x in range(len(ann_x)):
    if ann_y[x] != 'V' and ann_y[x] != 'x':
        continue
    a_points.append(ann_x[x])
    plt.annotate(ann_y[x], xy=(ann_x[x], data[ann_x[x]]), color='r')

for x in a_points:
    start = int(x/sub_len)
    start *= sub_len
    end = start + sub_len
    plt.plot(range(start, end), data[start:end], color='red', linewidth=4)

plt.title('(a)', loc='right')

plt.subplot(212)
score=score.repeat(sub_len)
plt.plot(range(length), [1-x for x in score], color='#9e026d')
plt.annotate('TSAX', xy=(0.8, 0.5))
plt.title('(b)', loc='right')
plt.tight_layout()
plt.show()



"""把rawdist转成异常分数
rawdist越大,越有可能是异常的
归一到(0,1)
"""

