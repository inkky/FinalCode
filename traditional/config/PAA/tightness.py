# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 21:28
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''

'''

from traditional.PAA.expand_PAA import BT_PAA, NT_PAA, Raw_PAA, EU
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from traditional.SAX.sax_variant import *

data = np.loadtxt('../../data/UCRtwoclass/ECG200.txt', delimiter=',')
labels = data[:, 0]
c = Counter(labels)
print(c)
b = zip(c.values(), c.keys())
c = list(sorted(b))
labels = np.where(labels == c[1][1], 1, 0)
data = data[:, 1:]
rows, cols = data.shape
k=3
alphabetSize=3
length=150

tight1 = list()
tight2 = list()
tight3 = list()
tight4 = list()
raw_dist,pred=EU(data,labels,k)
t = np.mean(raw_dist)
for i in range(1, length,8):
    t1, _ = esax(data, i, alphabetSize, k, labels)
    t2, _ = sax_td(data, i, alphabetSize, k, labels)
    t3, _ = NT_PAA(i, data,labels,k)
    t4, _ = Raw_PAA(i, data,labels,k)
    tight1.append(min(np.mean(t1) / t,1))
    tight2.append(np.mean(t2) / t)
    tight3.append(np.mean(t3) / t)
    tight4.append(np.mean(t4) / t)
print(tight1)
print(tight2)
print(tight3)
print(tight4)
plt.figure()
fig = plt.gcf()
fig.set_size_inches(8, 4)
x_new=range(1,length,8)
plt.plot(x_new,tight1, 'x-', label='ESAX')
plt.plot(x_new,tight2, '-', label='SAX_TD')
plt.plot(x_new,tight3, 'o-', label='OUR*')
plt.plot(x_new,tight4, 's-', label='PAA')
plt.legend(loc='upper right')
plt.xlim(1,length)
plt.xticks(np.arange(1,length,8))
# plt.ylim(0,1)
# plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('T(tightness)')
plt.xlabel('s')
plt.tight_layout()
plt.savefig('PAAimg/tight_NTpaa.png', dpi=300)
plt.show()
