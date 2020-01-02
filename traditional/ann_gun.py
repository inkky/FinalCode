# !/usr/bin/env python
# -*-coding:utf-8-*-


import numpy as np
from matplotlib import pyplot as plt
from EITable import *
from sklearn.preprocessing import minmax_scale, scale
from TSLSH import TSForest

fig, ax = plt.subplots()
fig.set_size_inches(10, 4)

obj = open('../codes/mitbih/data/ann_gun_CentroidA', 'r')
data_x = list()
data_y = list()
count = 0
for line in obj.readlines():
    tmp = line[:-1].split('  ')
    data_x.append(float(tmp[1]))
    data_y.append(float(tmp[2]))
    count += 1
    if count >= 9000:
        break
obj.close()

data_x = np.array(data_x)
data_y = np.array(data_y)
length = len(data_x)
sub_len = 150

data_x = data_y
new_x = data_x.reshape((length/sub_len, sub_len))
rows, cols = new_x.shape
new_x = scale(new_x, axis=1)

k = 10
new_xt = scale(new_x, axis=1)
t_data = preprocessing(new_xt, k)
predict = cal_score(new_xt, t_data, cols, k)
scores_EITable = list()
for x in predict:
    tmp = [x]*sub_len
    scores_EITable += tmp
scores_EITable = minmax_scale(scores_EITable)

forest = TSForest(tree_size=25)
new_x = scale(new_x, axis=1)
forest.fit(new_x)
predict = forest.predict()
scores_tslsh = list()
for x in predict:
    tmp = [x]*sub_len
    scores_tslsh += tmp
scores_tslsh = minmax_scale(scores_tslsh)



plt.figure(1)
plt.tight_layout()
# plt.xticks(range(0, 6000, sub_len))

plt.subplot(311)
plt.plot(range(length), data_x)
plt.plot(range(2*sub_len, 3*sub_len), data_x[2*sub_len:3*sub_len], color='r', linewidth=4)
plt.plot(range(14*sub_len, 17*sub_len), data_x[14*sub_len:17*sub_len], color='r', linewidth=4)
plt.plot(range(18*sub_len, 19*sub_len), data_x[18*sub_len:19*sub_len], color='r', linewidth=4)
plt.title('(a)', loc='right')

plt.subplot(312)
plt.plot(range(length), [1 - x for x in scores_EITable], color='#9e026d')
plt.annotate('EITable', xy=(0.8, 0.5))
plt.title('(b)', loc='right')

plt.subplot(313)
plt.plot(range(length), [1 - x for x in scores_tslsh], color='#cc2f69')
plt.annotate('DLDE', xy=(0.8, 0.5))
plt.title('(c)', loc='right')

# plt.show()
plt.savefig('../results/figures/ann_gun_y.png', dpi=300)