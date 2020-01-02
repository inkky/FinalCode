# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 17:05
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
PAA DRAW FIG
'''
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
import numpy as np
import matplotlib.pyplot as plt

#draw ecg200
data = np.loadtxt('data/ecg200.txt', delimiter=',')
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
data_score = scaler.fit_transform(data)
rows, cols = data.shape
# PAA transform (and inverse transform) of the data
n_paa_segments = 1
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(data))
a=np.mean(paa_dataset_inv.ravel())
print(a)

plt.figure(1)
fig = plt.gcf()
fig.set_size_inches(6, 3)
plt.plot(data_score[2].ravel(),"b-" ,label='Raw',linewidth=2.5,alpha=0.6)
plt.plot(paa_dataset_inv[2].ravel(),'r-',label='PAA',linewidth=2.5)
# print(data_score[2].ravel())
x_new=np.linspace(0,50)

plt.xticks(range(0,100,8))
plt.xlabel('time')
plt.ylabel('y')
plt.fill_between(x_new,a,paa_dataset_inv[2].ravel(),paa_dataset_inv[2].ravel()>=a, color='blue', alpha=.25)
# plt.title("TS1")
plt.tight_layout()
plt.legend(loc='upper center')
w = int((cols - 1) / n_paa_segments)
for i in range(n_paa_segments + 1):
    plt.axvline(x=w * i, ls='--', linewidth=0.5, color='k', alpha=0.5)
# plt.savefig('img/paa_example2.png', dpi=300)
plt.show()

