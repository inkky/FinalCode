"""
https://www.datascience.com/blog/python-anomaly-detection
https://www.kaggle.com/wittmaan/anomaly-detection-on-county-time-series
"""

import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

#load data
data=np.loadtxt('../data/others/sunspots.txt',float)
data_as_frame=pd.DataFrame(data,columns=['Month','SunSpots'])
data_month=data_as_frame["Month"]
data_SunSpots=data_as_frame["SunSpots"]
print(data_as_frame.head())

def moving_average(data,window_size):
    """
    Computes moving average
    using discrete linear convolution of two one dimensional sequences
    same’　返回的数组长度为max(M, N),边际效应依旧存在。
    需要定义一个N个周期的移动窗口
    ref:https://www.cnblogs.com/21207-iHome/p/6231607.html
    """
    #使用ones函数创建一个长度为N的元素均初始化为1的数组，然后对整个数组除以N，即可得到权重
    weight=np.ones(int(window_size))/float(window_size)
    return np.convolve(data,weight,'same')



def anomalies(data,window_size,sigma=1.0):
    """Helps in exploring the anamolies using stationary standard deviation
    """
    avg=moving_average(data,window_size).tolist()
    residual=data-avg
    std=np.std(residual)
    return


plt.figure()
plt.plot(data_as_frame["SunSpots"][1:100])
plt.show()