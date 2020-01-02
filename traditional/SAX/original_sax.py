import operator
import time
import sys
import matplotlib




matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
# from PAA.saxecg import toPAA, paa_inv, paa2letter, zscore, compareTS



class StringsAreDifferentLength(Exception): pass


def zscore(data):
    mu = np.mean(data)
    std = np.std(data)
    data_z = (data - mu) / std
    return data_z


def toPAA(d, win_size):
    '''paa segment'''
    paa = list()
    for i in range(0, len(d), win_size):
        high = min(len(d), i + win_size)
        segment = np.mean(d[i:high])
        # print(segment)
        paa.append(segment)

    return paa


def paa_inv(data, win_size):
    '''change mean into original length'''
    cols = len(data)
    data_inv = list()
    for i in range(len(data)):
        for j in range(win_size):
            data_inv.append(data[i])
    return data_inv

def breakTan(theta):
    angle_breakboints={''}

def breakSymble(alphabetSize):
    breakpoints = {'3': [-0.43, 0.43],
                   '4': [-0.67, 0, 0.67],
                   '5': [-0.84, -0.25, 0.25, 0.84],
                   '6': [-0.97, -0.43, 0, 0.43, 0.97],
                   '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                   '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                   '9': [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                   '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                   '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                   '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                   '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                   '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                   '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
                   '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
                   '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93,
                          1.19, 1.56],
                   '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97,
                          1.22, 1.59],
                   '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1,
                          1.25, 1.62],
                   '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52,
                          0.67, 0.84, 1.04, 1.28, 1.64]
                   }
    beta = breakpoints[str(alphabetSize)]
    return beta


def lookupTable(alphabetSize):
    beta = breakSymble(alphabetSize)
    lookup = np.zeros((alphabetSize, alphabetSize))
    for i in range(0, alphabetSize):
        for j in range(0, alphabetSize):
            if abs(i - j) <= 1:
                lookup[i][j] = 0
            else:
                high = np.max([i, j]) - 1
                low = np.min([i, j])
                lookup[i][j] = abs(beta[high] - beta[low])

    return lookup


def paa2letter(data, alphabetSize):
    '''
    sax
    change paa to letter
    '''
    alphabetizedX = ''
    beta = breakSymble(alphabetSize)
    aOffset = ord('a')
    for i in range(0, len(data)):
        letterFound = False
        for j in range(0, len(beta)):
            if np.isnan(data[i]):
                alphabetizedX += '-'
                letterFound = True
                break
            if data[i] < beta[j]:
                alphabetizedX += chr(aOffset + j)
                letterFound = True
                break
        if not letterFound:
            alphabetizedX += chr(aOffset + len(beta))

    return alphabetizedX


def compareTS(ts1, ts2, alphabetSize):
    """
        compare two string based on individual letter distance
        both strings have same length
        字符距离为之前breakpoint的距离，求和
        """
    if len(ts1) != len(ts2):
        raise StringsAreDifferentLength()
    mindist = 0
    letter_A = [x for x in ts1]
    letter_B = [x for x in ts2]

    lookup = lookupTable(alphabetSize)

    for i in range(len(letter_A)):
        mindist += lookup[ord(letter_A[i]) - 97][ord(letter_B[i]) - 97] ** 2
    mindist = np.sqrt(mindist)

    return mindist