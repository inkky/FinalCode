# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 14:15
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''

'''
import math
import random
import pickle
import itertools

import numpy as np
import pandas as pd
from sax_knn import sax
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,label_ranking_average_precision_score,label_ranking_loss,coverage_error

import matplotlib.pyplot as plt

np.random.seed(42)

df1=pd.read_csv('mitbih_test.csv',header=None)
df2=pd.read_csv('mitbih_train.csv',header=None)
df=pd.concat([df1,df2],axis=0)

print(df.head())
print(df.info())

#187列是label
df[187].value_counts() #计算series里面相同数据出现的频率

#设定 0 为 Normal
print(df[187].value_counts())

ECG=df.values #查看series的值
data=ECG[:,:-1]
label=ECG[:,-1].astype(int)
# print(data[1])
# print(label)

#返回非0的数组元组的索引，其中y是要索引数组的条件
C0 = np.argwhere(label == 0).flatten()
C1 = np.argwhere(label == 1).flatten()
C2 = np.argwhere(label == 2).flatten()
C3 = np.argwhere(label == 3).flatten()
C4 = np.argwhere(label == 4).flatten()

# tmp=np.hstack((data[C0, :][0],data[C1, :][0]))
# # print(tmp)
# plt.figure()
# plt.plot(tmp)
# plt.show()
#
#
# # PAA transform (and inverse transform) of the data
# n_paa_segments = 20
# paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
# paa_dataset_inv = paa.inverse_transform(paa.fit_transform(tmp))
# print(paa_dataset_inv.ravel())



# plt.figure() # Second, PAA
# fig = plt.gcf()
# fig.set_size_inches(6, 3)
# plt.plot(tmp.ravel(), "b-", alpha=0.4)
# plt.plot(paa_dataset_inv.ravel(), "b-")
# # plt.title("PAA")
# plt.tight_layout()
# plt.savefig('ecgMean.png', dpi=300)
# plt.show()

# exit()

# x = np.arange(0, 100)*8
x=np.arange(0,187)*8
plt.figure(figsize=(10,5))
# plt.plot(x, data[C0, :][0],linewidth = '2',color='blue', label="Cat. N")
# plt.plot(x, data[C1, :][0], color='red',label="Cat. S")
# plt.plot(x, data[C2, :][0], color='green',label="Cat. V")
# plt.plot(x, data[C3, :][0], color='black',label="Cat. F")
plt.plot(x, data[C4, :][0], color='grey',label="Cat. Q")
plt.legend()
plt.title("1-beat ECG for every category")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig('4.png')
plt.show()

exit()
x=np.arange(0,187)*8
plt.figure(figsize=(10,5))
plt.plot(x, data[C0, :][0],linewidth = '2')
plt.tight_layout()
plt.savefig('normalECG.png')
plt.show()



