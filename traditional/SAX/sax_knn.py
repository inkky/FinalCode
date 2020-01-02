# -*- coding: utf-8 -*-
# @Time    : 2018/6/13 15:01
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
ref:
https://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html
1.sax
2.sax-td: difference between begin point(end points) and average
3.extended sax: max and min points represented by symbol
4.tsax: trend( our method)
'''
import operator
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix


class StringsAreDifferentLength(Exception): pass


# ts1 = (2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34)
# ts2 = (0.50, 1.29, 2.58, 3.83, 3.25, 4.25, 3.83, 5.63, 6.44, 6.25, 8.75, 8.83, 3.25, 0.75, 0.72)
# l1 = len(ts1)
#
# data = np.loadtxt('sorted/Beef.txt', delimiter=',')
# labels = data[:, 0]
# data = data[:, 1:]
# rows, cols = data.shape


def zscore(data):
    """
    z_score norm
    """
    mu = np.mean(data)
    std = np.std(data)
    data_z = (data - mu) / std
    return data_z


def PAA(d, win_size):
    """
    paa segment
    """
    paa = list()
    for i in range(0, len(d), win_size):
        high = min(len(d), i + win_size)
        segment = np.mean(d[i:high])
        # print(segment)
        paa.append(segment)

    return paa


def paa_inv(data, win_size):
    """
    change mean into original length
    """
    cols = len(data)
    data_inv = list()
    # tmp = cols % win_size
    # print(len(data), tmp)
    for i in range(len(data)):
        for j in range(win_size):
            data_inv.append(data[i])
    # print(len(data_inv))
    # if tmp != 0:
    #     data_inv = data_inv[0:-(win_size - tmp)]
    return data_inv


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
    # print(beta)
    # exit()
    return beta


def lookupTable(alphabetSize):
    beta = breakSymble(alphabetSize)
    # print(beta)
    lookup = np.zeros((alphabetSize, alphabetSize))
    # print(lookup)
    # print(len(letter_A))
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
    """
    sax
    change paa to letter
    """
    alphabetizedX = ''
    beta = breakSymble(alphabetSize)
    aOffset = ord('a')
    # print(beta, aOffset)
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
    :param ts1:
    :param ts2:
    :return:
    """
    if len(ts1) != len(ts2):
        raise StringsAreDifferentLength()
    mindist = 0
    letter_A = [x for x in ts1]
    letter_B = [x for x in ts2]
    # print(letter_A)
    # print(letter_B)

    lookup = lookupTable(alphabetSize)
    # print(lookup)

    for i in range(len(letter_A)):
        mindist += lookup[ord(letter_A[i]) - 97][ord(letter_B[i]) - 97] ** 2
    mindist = np.sqrt(mindist)
    # print(mindist)
    return mindist


def sax(data, win_size, alphabetSize, k, label):
    # win_size = 8
    # alphabetSize = alphabetSize
    # k = 3

    # data = np.loadtxt('sorted/ECG200.txt', delimiter=',')
    # labels = data[:, 0]
    # data = data[:, 1:]
    rows, cols = data.shape
    # print(rows, cols)

    # plt.figure()
    # plt.plot(data[1], '-', label='raw data')

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
    tmax = list()  # extended sax 存放每段max的值
    tmin = list()  # extended sax 存放每段min的值
    sd=list() #sax_sd 存放均方差的值
    # print('1')

    for d in data_norm: # d 一条数据
        data_paa = PAA(d, win_size) #记录均值
        data_paa_inv = paa_inv(data_paa, win_size)

        # print(data_paa)
        # print(len(d)/win_size) # 分成的段数

        # paa2letter
        alpha = paa2letter(data_paa, alphabetSize)
        # print(len(alpha))
        # exit()

        paa.append(data_paa)
        paa_trans.append(data_paa_inv)
        paa_alpha.append(alpha)

        ########## SAX_SD ##############
        # GET the standard deviation of each segment
        # i = 0
        # tmpsd=list()
        # while i<len(d)/win_size:
        #     tmpsd.append(np.sqrt(np.var(d[i*win_size:(i+1)*win_size])))
        #     i=i+1
        #
        # # print(tmpsd)
        # # exit()
        # sd.append(tmpsd)



        ########## Extended SAX ##########
        # # max and min points
        # i = 0
        # tmpmax = list()
        # tmpmin = list()
        # while i < (len(d) / win_size):
        #     tmpmax.append(max(d[i*win_size:(i+1)*win_size]))
        #     tmpmin.append(min(d[i*win_size:(i+1)*win_size]))
        #     i = i + 1
        #
        # # paa2letter
        # alphamax = paa2letter(tmpmax, alphabetSize)
        # alphamin=paa2letter(tmpmin, alphabetSize)
        # # print('alphamax',alphamax,len(alphamax))
        # # print('alphamin', alphamin,len(alphamin))
        # tmax.append(alphamax)
        # tmin.append(alphamin)


        # ########## SAX_TD ###################
        # save the begin and end time point value
        i = 0
        tmpts = list()
        tmpte = list()
        # print(len(d))
        # print(win_size)
        while i < len(d) / win_size:
            # print(i)
            tmpts.append(d[i * win_size] - data_paa[i])
            tmpte.append(d[min((i + 1) * win_size,len(d)) - 1]- data_paa[i])
            i = i + 1

        # print(tmpts)
        # print(tmpte)
        # print(len(tmpte))
        ts.append(tmpts)
        te.append(tmpte)


        # ############# TSAX ##############
        # # save the relative binary trend
        # bit_tmp = ''
        # paa_tmp = list()
        #
        # for i in range(len(d)):
        #     if d[i] < data_paa_inv[i]:
        #         bit_tmp += '0'
        #     else:
        #         bit_tmp += '1'
        # # print(bit_tmxp)
        # # print(len(bit_tmp))
        # bit_data.append(int(bit_tmp, 2))



    paa_trans = np.array(paa_trans)
    paa = np.array(paa)
    paa_alpha = np.array(paa_alpha)
    ts = np.array(ts)
    te = np.array(te)
    sd=np.array(sd)
    tmax = np.array(tmax)
    tmin = np.array(tmin)

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
            # # print(raw_dist[i,j])
            raw_dist[i, j] = np.sqrt(win_size) * raw_dist[i, j]  # sax dist

            # ########### calculate bit distance ##########
            # c = bit_data[i] ^ bit_data[j]
            # ones = 0
            # while c:
            #     ones += 1
            #     c &= (c - 1)
            #
            # raw_dist[i, j] += (ones * 1.0 / win_size)*win_size  # bit dist

            ##### sax_td distance #####
            # raw_dist[i,j]+=np.sqrt(np.sum((np.square(ts[i]-ts[j])+np.square(te[i]-te[j]))/win_size)) #sax_td dist
            # raw_dist[i,j]+=np.sqrt(np.sum((np.square(ts[i]-ts[j])+np.square(te[i]-te[j])))) #sax_td dist
            raw_dist[i,j]+=np.sqrt(np.sum((np.square(ts[i]-ts[j])+np.square(te[i]-te[j])))) #sax_td dist

            ##### extended sax distance ########
            #
            #### raw_dist[i, j] += np.sqrt(np.sum(np.square(tmax[i] - tmax[j]) + np.square(tmin[i] - tmin[j])))
            # print('esax')
            # raw_dist[i,j]+=np.sqrt(win_size) * (compareTS(tmax[i],tmax[j],alphabetSize)+compareTS(tmin[i],tmin[j],alphabetSize))

            ##### sax_sd distance ########
            # print(sd)
            # raw_dist[i,j]+= np.sqrt(win_size) * np.sqrt(np.sum(np.square(sd[i]-sd[j])))



            raw_dist[j, i] = raw_dist[i, j]

        arg_index = np.argsort(raw_dist[i])
        tmp = dict()
        print(arg_index)
        if i in arg_index[:k]:
            print(i)
            for l in arg_index[:k + 1]:
                print(l)
                if l == i:
                    continue
                tmp[label[l]] = tmp.get(label[l], 0) + 1
                print(tmp)

        else:
            for j in arg_index[:k]:
                print('1',j)
                tmp[label[j]] = tmp.get(label[j], 0) + 1
                print(tmp)
        print(tmp)

        pre = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        print(pre)
        pred.append(pre)
    # print(pred)

    preanomaly = list()
    trueanomaly = list()
    for i in range(rows):
        if label[i] != 1.0:
            label[i] = 0
            trueanomaly.append(i)

        if pred[i] != 1.0:
            pred[i] = 0
            preanomaly.append(i)

    # print('win_size: %d, Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f,time:  %f' % \
    #       (win_size, precision_score(label, pred, average='macro'),
    #        recall_score(label, pred, average='macro'),
    #        f1_score(label, pred, average='macro'),
    #        roc_auc_score(label, pred, ),
    #        mean_squared_error(label, pred),
    #        (time.clock() - trainBeginTime)))
    # tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    # print('TN: %d,tp: %d,fn: %d,FP: %d', (tn, tp, fn, fp))
    # specificity = tn / (tn + fp)
    # falseAlarmRate = fp / (fp + tn)
    # print('specificity: %f,false alarm rate: %', (specificity, falseAlarmRate))
    # print('auc: %f, error rate: %f', (roc_auc_score(label, pred, ), mean_squared_error(label, pred)))

    # print which time point is anomaly
    # print(preanomaly)
    # print(trueanomaly)

    # plt.plot(paa_trans[1], '-', label='paa')
    # n_paa_segments = int(cols / win_size)
    # # print(n_paa_segments)
    # for i in range(n_paa_segments):
    #     plt.axvline(x=win_size * i, ls='--', linewidth=0.5, color='k', alpha=0.2)
    # plt.axvline(x=n_paa_segments * win_size-1, ls='--', linewidth=0.5, color='k', alpha=0.2)
    # plt.axvline(x=cols - 1, ls='--', linewidth=0.5, color='k', alpha=0.2)
    #
    # for i, txt in enumerate(paa_alpha[1]):
    #     # print(i, txt, data_paa_inv[i], len(paa_trans[1]))
    #     plt.annotate(txt, (i * win_size, paa_trans[1][i * win_size]))
    #
    # plt.legend(loc='best')
    # plt.show()

    return raw_dist, pred

    # ## ts1,ts2
    # plt.figure()
    # plt.plot(ts1, '-', label='raw data')
    #
    # # zscore
    # data_norm1 = zscore(ts1)
    # data_norm2 = zscore(ts2)
    # print(np.std(data_norm1))
    # plt.plot(data_norm1, '-', label='zscore data')
    #
    # # paa
    # data_paa = PAA(data_norm1, win_size)
    # data_paa_inv = paa_inv(data_paa, win_size)
    # # print(len(data_paa_inv))
    # plt.plot(data_paa_inv, '-', label='paa')
    # n_paa_segments = int(len(data_paa_inv) / win_size)
    # for i in range(n_paa_segments + 1):
    #     plt.axvline(x=win_size * i, ls='--', linewidth=0.5, color='k', alpha=0.2)
    # plt.axvline(x=len(data_paa_inv) - 1, ls='--', linewidth=0.5, color='k', alpha=0.2)
    #
    # data_paa2 = PAA(data_norm2, win_size)
    # data_paa_inv2 = paa_inv(data_paa2, win_size)
    #
    # # paa2letter
    # alpha1 = paa2letter(data_paa, alphabetSize)
    # for i, txt in enumerate(alpha1):
    #     # print(i,txt,data_paa_inv[i])
    #     plt.annotate(txt, (i * win_size + 1, data_paa_inv[i * win_size + 1]))
    #
    # alpha2 = paa2letter(data_paa2, alphabetSize)
    # print(alpha1, alpha2)
    #
    # # compare the two series
    # mindist = compareTS(alpha1, alpha2)
    # mindist = np.sqrt(win_size) * mindist
    # print('mindist=', mindist)
    #
    # plt.legend(loc='best')
    # plt.show()
    # # X = np.array([[ 1., -1.,  2.],
    # #                [ 2.,  0.,  0.],
    # #                [ 0.,  1., -1.]])
    # # X_scaled = preprocessing.scale(X,axis=1)
    # # print(X_scaled)
    # # print(X_scaled.std())


from collections import Counter

# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from tslearn.piecewise import PiecewiseAggregateApproximation

if __name__ == '__main__':
    print('(ones * 1.0 / win_size)*np.sqrt(win_size)')
    # win_size = 4
    alphabetSize = 3
    k = 3

    # file = os.listdir('twoclass/')
    # f = open('result/sax_auc.txt', 'a')
    # for filename in file:
    #     portion = os.path.splitext(filename)
    #     if portion[1] == ".txt":
    #         print(filename)
    #         data = np.loadtxt('sorted/'+filename, delimiter=',')
    #         sax(data, win_size, alphabetSize, k)



    data = np.loadtxt('../data/UCRtwoclass/BeetleFly.txt', delimiter=',')
    all_labels = data[:, 0]
    data = data[:, 1:]
    rows, cols = data.shape
    # print(rows,cols)
    #
    c = Counter(all_labels)
    # print(c)
    C0 = np.argwhere(all_labels == 1).flatten()
    C1 = np.argwhere(all_labels == 2).flatten()
    C2 = np.argwhere(all_labels == 3).flatten()
    C3 = np.argwhere(all_labels == 4).flatten()
    C4 = np.argwhere(all_labels == 5).flatten()
    #
    # print(C0)
    # print(C1)
    # print(C2)
    # print(C3)
    # print(C4)
    # exit()

    data_test=np.vstack((data[C0, :],data[C4, :]))
    label=np.hstack((all_labels[C0],all_labels[C4]))
    # print('04')

    # # print(label)
    # # print(data)

    # from sklearn.cross_validation import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)




    # x = np.arange(0, 140)
    # # x=np.arange(0,187)
    #
    # # PAA transform (and inverse transform) of the data
    # n_paa_segments = 10
    # paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    # paa_dataset_inv1 = paa.inverse_transform(paa.fit_transform(data[C0, :][0]))
    # paa_dataset_inv2 = paa.inverse_transform(paa.fit_transform(data[C4, :][0]))
    # plt.figure()  # Second, PAA
    # plt.plot(x, data[C0, :][0], "b-", alpha=0.4,label='N')
    # plt.plot(x,paa_dataset_inv1[0].ravel(), "b-")
    # plt.plot(x, data[C4, :][0], "r-", alpha=0.4,label='V')
    # plt.plot(x, paa_dataset_inv2[0].ravel(), "r-")
    # # plt.title("PAA")
    # # plt.legend(loc='upper right')
    # plt.tight_layout()
    # # plt.savefig('compareECG.png')
    # plt.show()
    # exit()
    #
    # plt.figure(figsize=(8, 3))
    # plt.plot(x, data[C0, :][0], linewidth='2', label="Cat. N")
    # plt.plot(x, data[C4, :][0]-6, linewidth='2',label="Cat. S")
    # plt.vlines(140, -5, 7, colors="c", linestyles="dashed")
    # plt.plot(x, data[C2, :][0]-6, label="Cat. V")
    # plt.plot(x, data[C3, :][0]-9, label="Cat. F")
    # plt.plot(x, data[C4, :][0]-12, label="Cat. Q")
    # plt.legend(loc='upper right')
    # plt.title("1-beat ECG for every category")
    # plt.xlabel("Time")
    # plt.tight_layout()
    # # plt.savefig('twoECG.png')
    # plt.show()

    # exit()
    # print(rows, cols)


    # winsize
    for win_size in [4]:
        # print('win_size:',win_size)
        rawdist, preanomaly, trueanomaly = sax(data, win_size, alphabetSize, k, all_labels)
    #     # print(rawdist)
    #     # print(preanomaly)
    #     # print(trueanomaly)

    # win_size = 4
    # rawdist, preanomaly, trueanomaly = sax(data_test, win_size, alphabetSize, k, label)
    # print(rawdist)
    # print(preanomaly)
    # print(trueanomaly)

    # plt.figure()
    # fig = plt.gcf()
    # fig.set_size_inches(8, 4)
    # plt.subplot(211)
    # x = np.arange(0, 140)
    # datatmp=data.flatten()
    #
    # # plt.title('TSAX Detected anomaly ECG beats')
    # plt.plot(datatmp[0:1500],color='lightblue')
    # for i in range(0, 10):
    #     if i in preanomaly:
    #         plt.plot(x + i * cols, data[i], color='red')
    #     # else:
    #     #     plt.plot(x + i * cols, data[i], color='lightblue')
    # # plt.annotate('anomaly', xy=(400, 1), xytext=(400, 2.5),
    # #              arrowprops=dict(facecolor='black', shrink=0.1, headlength=3))
    #
    # plt.subplot(212)
    # # plt.title('True anomaly ECG beats')
    # for i in range(0, 10):
    #     if i in trueanomaly:
    #         plt.plot(x + i * cols, data[i], color='red')
    #     else:
    #         plt.plot(x + i * cols, data[i], color='lightblue')
    #
    # plt.xlabel('Time')
    # # plt.annotate('anomaly', xy=(400, 1), xytext=(400, 2.5),
    # #              arrowprops=dict(facecolor='black', shrink=0.1, headlength=3))
    # plt.savefig('img/ECG_TSAX.png', dpi=300)
    # plt.show()
