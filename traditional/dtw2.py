# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 16:14
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''

'''

import numpy as np
import sys
from sklearn.metrics.pairwise import manhattan_distances,euclidean_distances
#一维数组曼哈顿距离，二维数组欧式距离
def distance(x,y):
    return abs(x-y)

################################# DTW ##################################
'''
算法复杂度为O(N2)
'''
def dtw(X,Y):
    #生成原始距离矩阵
    # M = [[distance(X[i], Y[i]) for i in range(len(X))]for j in range(len(Y))]
    # M=[[manhattan_distances(X[i],Y[i]) for i in range(len(X))] for j in range(len(Y))]
    l1 = len(X)
    l2 = len(Y)
    M=np.zeros(l1,l2)
    for i in range(l1):
        for j in range(l2):
            M[i,j]=manhattan_distances(X[i],Y[j])
    print('M',M)


    D = [[0 for i in range(l1 + 1)] for i in range(l2 + 1)]
    # D[0][0] = 0
    for i in range(1, l1 + 1):
        D[0][i] = sys.maxsize
    for j in range(1, l2 + 1):
        D[j][0] = sys.maxsize

    #动态计算最短距离矩阵
    for j in range(1, l2 + 1):
        for i in range(1, l1 + 1):
            D[j][i] = distance(X[i -1],Y[j-1 ])+ min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1])
            # D[j][i] = M[i-1,j-1]+ min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1])
            print(j,i,D[j][i])
    print(D)
    return D


################################# FastDTW ##################################
# FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador, Philip Chan.
'''
限制和数据抽象
O(N)
'''
from fastdtw import  fastdtw
from scipy.spatial.distance import euclidean



if __name__ == '__main__':
    X = [1, 2, 3, 4]
    Y = [1, 2, 7, 4, 5]
    Distance_dtw=dtw(X,Y)
    Distance_fastdtw,path=fastdtw(X,Y,dist=euclidean)
    print(Distance_fastdtw)
    print(path)


