import os
import numpy
import operator
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale, StandardScaler
from matplotlib import pyplot as plt
from expand_PAA import BT_PAA, Raw_PAA, NT_PAA, Cosine_PAA


def RawEuclidean(filename):
    # data = numpy.loadtxt('data/beef.txt', delimiter=',')
    data = numpy.loadtxt('sorted/' + filename, delimiter=',')
    labels = data[:, 0]
    data = data[:, 1:]
    rows, cols = data.shape

    k = 3
    trainBeginTime = time.clock()
    raw_dist = numpy.zeros((rows, rows))
    pred = list()
    for i in range(rows):
        for j in range(i + 1, rows):
            # raw_dist[i, j] = numpy.linalg.norm(data[i] - data[j])
            raw_dist[i, j] = numpy.sum(numpy.square(data[i] - data[j]))
            raw_dist[i, j] **= 0.5

            raw_dist[j, i] = raw_dist[i, j]

        arg_index = numpy.argsort(raw_dist[i])
        tmp = dict()
        if i in arg_index[:k]:
            for l in arg_index[:k + 1]:
                if l == i:
                    continue
                tmp[labels[l]] = tmp.get(labels[l], 0) + 1
        else:
            for l in arg_index[:k]:
                tmp[labels[l]] = tmp.get(labels[l], 0) + 1

        pre = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        pred.append(pre)

    for i in range(rows):
        if labels[i] != 1:
            labels[i] = 0

        if pred[i] != 1.0:
            pred[i] = 0

    # print('lable',labels)
    # print('pred',pred)

    # print(filename)
    print('Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f,time:  %f' % \
          (precision_score(labels, pred, average='macro'),
           recall_score(labels, pred, average='macro'),
           f1_score(labels, pred, average='macro'),
           roc_auc_score(labels, pred, ),
           mean_squared_error(labels, pred),
           (time.clock() - trainBeginTime)))

    f.write(filename)
    f.write(',%f,%f,%f,%f,%f' % \
            (precision_score(labels, pred, average='macro'), recall_score(labels, pred, average='macro'),
             f1_score(labels, pred, average='macro'), roc_auc_score(labels, pred, ),
             mean_squared_error(labels, pred),))
    f.write('\n')

    # f.write(filename)
    # f.write(', Precision: %f, Recall: %f, F1: %f, auc:  %f, time cost:  %f' % \
    #         (precision_score(labels, pred, average='macro'),
    #          recall_score(labels, pred, average='macro'),
    #          f1_score(labels, pred, average='macro'),
    #          roc_auc_score(labels, pred),
    #          (time.clock() - trainBeginTime)))
    # f.write('\n')

    # f.write(filename + '\n')
    # f.write('Precision: %f \n' % precision_score(labels, pred, average='macro'))
    # f.write('Recall   : %f \n' % recall_score(labels, pred, average='macro'))
    # f.write('F1       : %f \n' % f1_score(labels, pred, average='macro'))
    # f.write('AUC      : %f \n' % roc_auc_score(labels, pred))
    # f.write('COST TIME     : %f \n' % (time.clock() - trainBeginTime))
    # f.write('\n')
    return raw_dist


file = os.listdir('sorted/')
f = open('result/knn_Cosin_error.txt', 'a')
for filename in file:
    portion = os.path.splitext(filename)
    if portion[1] == ".txt":
        print(filename)
        RawEuclidean(filename)

# dist=RawEuclidean('CBF.txt')
f.close()
# print(numpy.mean(dist))
