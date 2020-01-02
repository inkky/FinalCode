# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 11:38
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
knn time cost
'''
import os
import sys
import numpy as np
import operator
import random
from numpy import array, sum, sqrt
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
# from bitarray import bitarray
from matplotlib import pyplot as plt

from expand_PAA import BT_PAA, NT_PAA, Raw_PAA
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.loadtxt('../../data/UCRtwoclass/Coffee.txt', delimiter=',')
# data4 = np.loadtxt('sorted/Plane.txt', delimiter=',')
labels = data[:, 0]
c = Counter(labels)
print(c)
b = zip(c.values(), c.keys())
c = list(sorted(b))
labels = np.where(labels == c[1][1], 1, 0)
data = data[:, 1:]

k=3

T1 = list()
T2 = list()
T3 = list()

for i in range(2, 11):
    _, time_list1 = BT_PAA(i,data,labels,k)
    _, time_list2 = NT_PAA(i,data,labels,k)
    _, time_list3 = Raw_PAA(i,data,labels,k)
    T1.append(time_list1)
    T2.append(time_list2)
    T3.append(time_list3)
# print(T1,T2,T3)
print(T1)

plt.figure()
fig = plt.gcf()
width=0.15
fig.set_size_inches(6, 3)
x_new=np.arange(2,11)

plt.bar(x_new,T1,width,label='BT_PAA')
plt.bar(x_new+width,T2,width,label='NT_PAA')
plt.bar(x_new+2*width,T3,width,label='PAA')

# plt.legend(loc='upper right')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=5)
# plt.xlim(2,4)
plt.xticks(np.arange(2,11,1))
# plt.ylim(0,1)
# plt.yticks(np.arange(0,1.1,0.1))
plt.title('Beef')
# plt.annotate('PAA',xy=(2,0.1),xytext=(2,0.2),arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.ylabel('cost time')
plt.xlabel('s')
plt.tight_layout()
plt.savefig('img/time_Coffee.png', dpi=300)
plt.show()