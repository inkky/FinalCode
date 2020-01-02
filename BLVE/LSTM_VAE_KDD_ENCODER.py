# -*- coding: utf-8 -*-
# @Author: cyy
# @Date  : 2019/3/18

# -*- coding: utf-8 -*-
"""
One simple Implementation of LSTM_VAE based algorithm for Anomaly Detection in Multivariate Time Series;

Author: Schindler Liang

Reference:
    https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
    https://github.com/twairball/keras_lstm_vae
    https://arxiv.org/pdf/1711.00614.pdf
"""
import numpy as np
import tensorflow as tf
from read_file_kdd import Data_Hanlder
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell
from sklearn.metrics import roc_auc_score, precision_score, classification_report,recall_score, f1_score, mean_squared_error,precision_recall_fscore_support,confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
import pickle
import json
import os
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x)


# 创建多层LSTM
def _LSTMCells(unit_list, act_fn_list):
    return MultiRNNCell([LSTMCell(unit,
                                  activation=act_fn)
                         for unit, act_fn in zip(unit_list, act_fn_list)])


class LSTM_VAE(object):
    def __init__(self, dataset_name, config):
        self.outlier_fraction = config['outlier_fraction']
        self.data_source = Data_Hanlder(dataset_name, config)

        self.n_hidden = config['n_hidden']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.train_iters = config['train_iters']

        self.input_dim = self.data_source.cols #输入的序列长度
        self.z_dim = config['z_dim']
        self.time_steps = config['time_steps']

        # self.pointer = 0
        self.anomaly_score = 0  # 阈值
        self.sess = tf.Session(config=config_gpu)
        self._build_network()
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        print('build network ing...')
        # lstm expects the input data (X) : [samples, time steps, features]
        with tf.variable_scope('ph'): #placeholder
            self.X = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.input_dim], name='input_X')

        with tf.variable_scope('encoder'):
            # 均值
            with tf.variable_scope('lat_mu'):
                mu_fw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])
                mu_bw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])

                (mu_fw_outputs, mu_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    mu_fw_lstm_cells,
                    mu_bw_lstm_cells,
                    self.X, dtype=tf.float32)
                self.mu_outputs = tf.add(mu_fw_outputs, mu_bw_outputs)

            #方差
            with tf.variable_scope('lat_sigma'):
                sigma_fw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                sigma_bw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                (sigma_fw_outputs, sigma_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    sigma_fw_lstm_cells,
                    sigma_bw_lstm_cells,
                    self.X, dtype=tf.float32)
                sigma_outputs = tf.add(sigma_fw_outputs, sigma_bw_outputs)

                #sampleing gussian
                sample_Z = self.mu_outputs + sigma_outputs * tf.random_normal(
                    tf.shape(self.mu_outputs),
                    0, 1, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            recons_lstm_cells = _LSTMCells([self.n_hidden, self.input_dim], [lrelu, lrelu])
            self.recons_X, _ = tf.nn.dynamic_rnn(recons_lstm_cells, sample_Z, dtype=tf.float32)

        with tf.variable_scope('re_encoder'):
            with tf.variable_scope('re_lat_mu'):
                re_mu_fw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])
                re_mu_bw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])
                (re_mu_fw_outputs, re_mu_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    re_mu_fw_lstm_cells,
                    re_mu_bw_lstm_cells,
                    self.recons_X, dtype=tf.float32)
                re_mu_outputs = tf.add(re_mu_fw_outputs, re_mu_bw_outputs)

            with tf.variable_scope('re_lat_sigma'):
                re_sigma_fw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                re_sigma_bw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                (re_sigma_fw_outputs, re_sigma_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    re_sigma_fw_lstm_cells,
                    re_sigma_bw_lstm_cells,
                    self.recons_X, dtype=tf.float32)
                re_sigma_outputs = tf.add(re_sigma_fw_outputs, re_sigma_bw_outputs)


        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1, tf.keras.backend.ndim(self.X))
            # reconstruction loss
            self.recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)

            # re-encoder loss
            self.enc_loss = tf.losses.mean_squared_error(sigma_outputs,
                                                         re_sigma_outputs) + tf.losses.mean_squared_error(self.mu_outputs,
                                                                                                          re_mu_outputs)

            # KL divergence
            self.kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_outputs - tf.square(self.mu_outputs) - tf.exp(sigma_outputs))

            # >>>>>> minimize loss
            self.opt_loss = self.recons_loss + self.kl_loss + self.enc_loss
            # self.opt_loss = self.recons_loss + self.kl_loss

            # >>>>>> anomaly score
            self.point_loss = tf.square(self.X - self.recons_X)
            # self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X), reduction_indices=reduce_dims)

            self.all_losses = 0.6*tf.reduce_sum(tf.square(self.X - self.recons_X),
                                            reduction_indices=reduce_dims) + \
                              0.4*tf.reduce_sum(tf.square(sigma_outputs - re_sigma_outputs) + \
                                            tf.square(self.mu_outputs - re_mu_outputs),
                reduction_indices=reduce_dims)


        # with tf.variable_scope('loss'):
        #     reduce_dims = np.arange(1, tf.keras.backend.ndim(self.X))
        #     self.recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
        #     # KL divergence
        #     self.kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_outputs - tf.square(mu_outputs) - tf.exp(sigma_outputs))
        #
        #     self.opt_loss = self.recons_loss + self.kl_loss
        #     # todo: what is all_loss
        #     self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X), reduction_indices=reduce_dims)

        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)

    def train(self):
        for i in tqdm(range(self.train_iters),dynamic_ncols=True):
            this_X = self.data_source.fetch_data(self.batch_size)#一个batch_size的train
            # print(this_X.shape)
            # print(self.sess.run([self.recons_loss],feed_dict={self.X:this_X}))
            # print(self.sess.run([self.kl_loss],feed_dict={self.X:this_X}))

            self.sess.run([self.uion_train_op], feed_dict={
                self.X: this_X
            })

            if i % 2000 == 0:
                # mse均方误差
                mse_loss = self.sess.run([self.opt_loss], feed_dict={
                    self.X: self.data_source.train
                })
                tqdm.write('round {}: with loss: {}'.format(i, mse_loss))
                # print('round {}: with loss: {}'.format(i, mse_loss))

        # self._arange_score(self.data_source.train)


    # 计算异常得分阈值
    def _arange_score(self, input_data): #todo:异常得分归一化
        #计算真实的异常比例
        # print("test instance",self.data_source.test.shape)
        c=Counter(self.data_source.test_label)
        self.true_ano=c[-1]/(c[1]+c[-1])
        print('true_ano',self.true_ano)


        # 相当于测试集
        input_all_losses = self.sess.run(self.all_losses, feed_dict={
            self.X: input_data
        })
        # input_all_losses=(input_all_losses-np.min(input_all_losses))/(np.max(input_all_losses)-np.min(input_all_losses))
        #归一化
        # self.anomaly_score = np.percentile(input_all_losses, (1 - self.outlier_fraction)*100)  # 小于这个值的观察值占总数q的百分比 * 100)
        # print(input_all_losses)
        self.anomaly_score=np.percentile(input_all_losses,(1-self.true_ano)*100)
        # self.anomaly_score=(self.anomaly_score-min(input_all_losses))/(max(input_all_losses)-min(input_all_losses))
        # print(self.anomaly_score)





    # 统计测试机的异常得分并分标签1,-1
    def judge(self, test,label):
        test2=np.split
        self.all_test_loss = self.sess.run(self.all_losses, feed_dict={
            self.X: test
        })

        # 画出潜在空间的点的可视化
        z_mu=self.sess.run(self.mu_outputs,feed_dict={self.X:test})
        print(z_mu.shape)

        # 2D
        # z_mu = np.reshape(z_mu, (-1, 2))
        # print(z_mu.shape)
        # plt.figure()
        # plt.scatter(z_mu[:,0],z_mu[:,1],c=label)
        # # plt.colorbar()
        # # plt.grid()
        # plt.legend('x1')
        # plt.show()
        # plt.savefig('img_lat/lat_represent.png')

        # 3D
        # z_mu = np.reshape(z_mu, (-1, 3))
        # print(z_mu.shape)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # print(label)
        # for u in range(len(label)):
        #     if label[u]==1:
        #         g1=ax.scatter(z_mu[u,2],z_mu[u,1],z_mu[u,0],c='b')
        #     elif u<6000:
        #         g2=ax.scatter(z_mu[u, 2], z_mu[u, 1], z_mu[u, 0], c='r',marker='^')
        # # ax.scatter(z_mu[:,0],z_mu[:,1],z_mu[:,2],c=label)
        # plt.legend(handles=[g1, g2], labels=['normal', 'anomaly'], loc='upper right')
        # # ax.view_init(20, 200)  # 第一个参数是竖直旋转，第二个参数是水平旋转，旋转单位是度°.初始角度是ax.view_init(30, -60)
        # plt.savefig('img_lat/lat_kdd5.png',dpi=300)
        #
        # plt.show()



        # # 计算每个时间点的异常得分
        # rec_t = self.sess.run([self.recons_X], feed_dict={self.X: test})
        # rec_t = np.array(rec_t).reshape((-1, 121))
        #
        # test_t = test.reshape((-1, 121))
        # # num=100
        # for num in [2, 3, 101, 111, 121, 141, 142, 151, 152, 153, 154, 155, 156, 157,100500,111500,122501,122502,123503,123204]:
        #     print(self.data_source.test_label[num])
        #     plt.figure()
        #     # plt.plot(test_t[1])
        #     plt.plot(rec_t[num] / max(rec_t[num]))
        #     plt.savefig('img_kdd/rec' + str(num) + '.png', bbox_inches = 'tight',dpi=300)
        #     plt.close()
        #     plt.clf()
        #
        #     plt.figure()
        #     plt.plot(test_t[num])
        #     plt.savefig('img_kdd/test' + str(num) + '.png',bbox_inches = 'tight', dpi=300)
        #     # plt.show()
        #     plt.close()
        #     plt.clf()

        # all_test_loss=(all_test_loss-min(all_test_loss))/(max(all_test_loss)-min(all_test_loss))
        self._arange_score(test)

        # print(all_test_loss)
        # result=np.where(all_test_loss<1,1,-1)
        # print(self.anomaly_score)
        result = map(lambda x: 1 if x < self.anomaly_score else -1, self.all_test_loss)
        # print(list(result))

        # todo:result的输出
        return list(result)


    def plot_confusion_matrix(self):
        """
        data_source.test_label:测试机的label
        predict_label：预测的label
        :return:
        """
        # print("original_label:", self.data_source.test_label)
        predict_label = self.judge(self.data_source.test,self.data_source.test_label)
        # print('test shape',self.data_source.test.shape)
        # print("predict_label:", predict_label)
        # self.data_source.plot_confusion_matrix(self.data_source.test_label, predict_label, ['Abnormal', 'Normal'],
        #                                        'LSTM_VAE Confusion-Matrix')


        # #混淆矩阵
        # confusion=confusion_matrix(self.data_source.test_label,predict_label)
        # print(confusion)
        #
        # print(classification_report(self.data_source.test_label,predict_label))


        # 效率
        print('Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f' % \
              (precision_score(self.data_source.test_label, predict_label, average='macro'),
               recall_score(self.data_source.test_label, predict_label, average='macro'),
               f1_score(self.data_source.test_label, predict_label, average='macro'),
               roc_auc_score(self.data_source.test_label, predict_label),
               mean_squared_error(self.data_source.test_label, predict_label),
               ))

        # precision, recall, f1, _ = precision_recall_fscore_support(self.data_source.test_label,
        #                                                            predict_label,
        #                                                            average='macro')
        # print("Prec = %.4f | Rec = %.4f | F1 = %.4f"
        #       % (precision, recall, f1))


        return self.data_source.test_label, predict_label,self.all_test_loss,self.true_ano


def main():
    # Hyperparameters
    config = dict()
    config['num_layers'] = 3  # number of layers of stacked RNN's default=3
    config['hidden_size'] = 60  # memory cells in a layer default=60
    config['num_lattern'] = 20  # number of units in the latent space default=20

    config['time_steps'] = 1 #default=1
    config['z_dim'] = 2  # 潜在空间维度 default=32； 如果要看latent space，z_dim=2or3
    config['n_hidden'] = 8 #todo：？ default=8
    config['batch_size'] = 50  # 批梯度下降算法每次迭代都遍历批中的所有样本 default=50
    config['learning_rate'] = 1e-5  # 学习率 default=1e-5
    config['train_iters'] = 8000  # 1个iteration等于使用batchsize个样本训练一次 default=180000
    config['num_layers'] = 3  # number of layers of stacked RNN's default=3
    config['if_train'] = True #default=True
    config['epoch'] = 1000  # 1个epoch等于使用训练集中的全部样本训练一次 default=1000
    config['outlier_fraction'] = 0.5

    # #保存config
    # with open('config.pickle','wb') as handle:
    #     pickle.dump(config,handle,protocol=pickle.HIGHEST_PROTOCOL)
    #     print('save seccess')

    # #读取config
    # with open('config.pickle','rb') as handle:
    #     config=pickle.load(handle)

    with open('result/lstm_vae_encoder_kdd64.txt', 'a') as result:
        lstm_vae=LSTM_VAE('kdd',config)
        lstm_vae.train()
        test_label, predict_label,loss,ano_ratio = lstm_vae.plot_confusion_matrix()
        auc = roc_auc_score(test_label, predict_label, )
        if auc < 0.5:
            auc = 1 - auc
        result.write('auc:  %f, error: %f \n' %
                     (auc, mean_squared_error(
                         test_label, predict_label),))
        # result.write('Precision: %f, Recall: %f, F1: %f,auc:  %f,error: %f \n' %
        #              (precision_score(
        #                  test_label, predict_label, average='macro'), recall_score(
        #                  test_label, predict_label, average='macro'), f1_score(
        #                  test_label, predict_label, average='macro'), auc, mean_squared_error(
        #                  test_label, predict_label),))
        js = json.dumps(config)
        result.write(js + '\n')
        result.write('loss '+str(loss)+'\n')
        result.write('anomaly ratio '+str(ano_ratio)+'\n')





if __name__ == '__main__':
    main()

