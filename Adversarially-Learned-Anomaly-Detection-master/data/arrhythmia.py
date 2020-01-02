import logging
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''
Arrhythmia
This database contains 279 attributes, 206 of which are linear valued and the rest are nominal. 
addredd: http://archive.ics.uci.edu/ml/datasets/arrhythmia
Number of Instances: 452
Number of Attributes: 279

Class code :          Class   :                  Number of instances:
       01             Normal				                      245
       02             Ischemic changes (Coronary Artery Disease)   44
       03             Old Anterior Myocardial Infarction           15
       04             Old Inferior Myocardial Infarction           15
       05             Sinus tachycardy			                   13
       06             Sinus bradycardy			                   25
       07             Ventricular Premature Contraction (PVC)       3
       08             Supraventricular Premature Contraction	    2
       09             Left bundle branch block 		                9	
       10             Right bundle branch block		               50
       11             1. degree AtrioVentricular block	            0	
       12             2. degree AV block		                    0
       13             3. degree AV block		                    0
       14             Left ventricule hypertrophy 	                4
       15             Atrial Fibrillation or Flutter	            5
       16             Others				                       22


In this experiment:
For the arrhythmia dataset, anomalous classes
represent 15% of the data and therefore the 15% of samples
with the highest anomaly scores are likewise classified as
anomalies (positive class). v

In this paper:
然而所有数据集已经经过处理变成0,1，0为正常386，1为异常66
做不出paper的结果，太奇怪了
'''

logger = logging.getLogger(__name__)


def get_train(label=0, scale=False, *args):
    """Get training dataset for Thyroid dataset
    这个label没有被用到
    """
    return _get_adapted_dataset("train", scale)


def get_test(label=0, scale=False, *args):
    """Get testing dataset for Thyroid dataset"""
    return _get_adapted_dataset("test", scale)


def get_valid(label=0, scale=False, *args):
    """Get validation dataset for Thyroid dataset"""
    return None


def get_shape_input():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274)  # 274个Attributes


def get_shape_input_flatten():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274)


def get_shape_label():
    """Get shape of the labels in Thyroid dataset"""
    return (None,)


def get_anomalous_proportion():
    return 0.15  # 异常数据的比例


def _get_dataset(scale):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    data = scipy.io.loadmat("data/arrhythmia.mat")

    full_x_data = data["X"]  # (452, 274)
    full_y_data = data['y']  # (452, 1)

    from collections import Counter
    # print(Counter(full_y_data.flatten().astype(int)))
    # exit()


    # #》》》》》》》》》》》不同异常比例
    # full_y_data = full_y_data.flatten().astype(int)
    # normal_data = full_x_data[full_y_data == 0]
    # abnormal_data = full_x_data[full_y_data == 1]
    # normal_label = full_y_data[full_y_data == 0]
    # abnormal_label = full_y_data[full_y_data == 1]

    # x_train, x_test, y_train, y_test = train_test_split(normal_data, normal_label, test_size=0.25, random_state=0)

    # print('ratio=0.4')
    # x_test=np.concatenate([x_test,abnormal_data],axis=0)
    # y_test=np.concatenate([y_test,abnormal_label],axis=0)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # y_test = y_test.flatten().astype(int)

    # print('ratio=0.3')
    # x_test = np.concatenate([x_test, abnormal_data[:41]], axis=0)
    # y_test = np.concatenate([y_test, abnormal_label[:41]], axis=0)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # y_test = y_test.flatten().astype(int)

    # print('ratio=0.2')
    # x_test = np.concatenate([x_test, abnormal_data[:24]], axis=0)
    # y_test = np.concatenate([y_test, abnormal_label[:24]], axis=0)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # y_test = y_test.flatten().astype(int)

    # print('ratio=0.1')
    # x_test = np.concatenate([x_test, abnormal_data[:11]], axis=0)
    # y_test = np.concatenate([y_test, abnormal_label[:11]], axis=0)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # y_test = y_test.flatten().astype(int)



    #>>>>>>>>>>>原始测试
    # 50% 分为测试集
    x_train, x_test, \
    y_train, y_test = train_test_split(full_x_data,
                                       full_y_data,
                                       test_size=0.5,
                                       random_state=42)# default=0.6,随机数种子是为了保证每次随机的结果都是一样的

    y_train = y_train.flatten().astype(int) #转换数据类型
    y_test = y_test.flatten().astype(int)






    if scale:
        print("Scaling dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset


def _get_adapted_dataset(split, scale):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale)
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    print("Size of split", split, ":", dataset[key_lbl].shape[0])
    return (dataset[key_img], dataset[key_lbl])


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)


if __name__ == '__main__':
    _get_dataset(True)
