# !/usr/bin/env python
# -*-coding:utf-8-*-


import numpy as np
from matplotlib import pyplot as plt
from EITable import *
from sklearn.preprocessing import minmax_scale
from TSLSH import TSForest

fig, ax = plt.subplots()
fig.set_size_inches(10, 4)
# ax.grid(True, linestyle='-.', axis='x')


d_file = '../codes/mitbih/data/mitdbx108.txt'
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

d_file = '../codes/mitbih/data/mitdbx108_annotations.txt'
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

sub_len = 360
data = data[:length]
data = np.array(data)
data = data[:, 0]
ndata = data.reshape((int(length/sub_len), sub_len))

rows, cols = ndata.shape

# EITABLE
k = 3
tdata = scale(ndata, axis=1)
t_data = preprocessing(tdata, k)
predict = cal_score(tdata, t_data, cols, k)
scores_eitable = list()
for x in predict:
    tmp = [x]*sub_len
    scores_eitable += tmp
scores_eitable = minmax_scale(scores_eitable)
print(scores_eitable)
# print(np.where(scores_eitable == 0 ))
# exit()

# TSLSH
# forest = TSForest(tree_size=25)
# ndata = scale(ndata, axis=1)
# forest.fit(ndata)
# predict = forest.predict()
# predict = minmax_scale(predict)
# scores_tslsh = list()
# for x in predict:
#     tmp = [x]*sub_len
#     scores_tslsh += tmp
# # scores_tslsh = minmax_scale(scores_tslsh)


plt.tight_layout()
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
    plt.plot(range(start, end), data[start:end], color='R', linewidth=4)

plt.title('(a)', loc='right')

plt.subplot(212)
plt.plot(range(length), [1 - x for x in scores_eitable], color='#9e026d')
plt.annotate('EITable', xy=(0.8, 0.5))
plt.title('(b)', loc='right')

# plt.subplot(313)
# plt.plot(range(length), [1-x for x in scores_tslsh], color='#cc2f69')
# plt.annotate('DLDE', xy=(0.8, 0.5))
# plt.title('(c)', loc='right')

plt.show()
# plt.savefig('../results/figures/mitbih108.png', dpi=300)