#  #################################################################
#  This file contains the main LyDROO operations, including building convolutional DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- January 2021. Written based on Tensorflow 2 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from system_params import * 

from numpy.random import default_rng
rng = default_rng()

# DNN network for memory

class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10, 
        batch_size=100, 
        memory_size=1000,
        test_size=32,
        output_graph=False
    ):

        self.net = net  # the size of the DNN
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.test_size = test_size
        
        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []
        self.val_cost_his = []


        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        scaled_kn_size = 2
        kn_size = 4
        self.model = keras.Sequential([
                    layers.Conv1D(32, kn_size, activation='relu',input_shape=[int(self.net[0]/no_nn_inputs),kn_size]), # first Conv1D with 32 channels and kearnal size 3
                    layers.Conv1D(64, 3, activation='relu'), # second Conv1D with 32 channels and kearnal size 3
                    layers.Conv1D(64, 3, activation='relu'), # second Conv1D with 32 channels and kearnal size 3
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(self.net[-1], activation='sigmoid')
                    # layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
                    # layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
                    # layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
                ])
# 
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size + self.test_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size + self.test_size)
        batch_memory = self.memory[sample_index, :]



        
        h_train = batch_memory[:, 0: self.net[0]]
        h_train = h_train.reshape(self.batch_size+self.test_size,int(self.net[0]/no_nn_inputs),no_nn_inputs)
        m_train = batch_memory[:, self.net[0]:]

        
        # print(h_train)          # (128, 10)
        # print(m_train)          # (128, 10)

        # train the DNN
        hist = self.model.fit(h_train, m_train, verbose=0, validation_split=1/3)
        self.cost = hist.history['loss'][0]
        self.val_cost = hist.history['val_loss'][0]
        
        assert(self.cost > 0)

        self.cost_his.append(self.cost)
        self.val_cost_his.append(self.val_cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into tf placeholder
        h = h.reshape(no_users, no_nn_inputs)
        h = h[np.newaxis, :]

        m_pred = self.model.predict(h)
        m_list = []

        if mode == 'OP':
            m_list = self.knm(m_pred[0], k).copy()
        elif mode =='KNN':
            m_list =  self.knn(m_pred[0], k).copy()
        elif mode =='OPN':
            m_list =  self.opn(m_pred[0], k).copy()
        else:
            print("The action selection must be 'OP' or 'KNN'")
        return np.unique(m_list, axis=0)
    
    # def knm(self, m, k = 1):
    #     # return k order-preserving binary actions
    #     m_list = []
    #     # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
    #     m_list.append(1*(m>0.5))
        
    #     if k > 1:
    #         # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
    #         m_abs = abs(m-0.5)
    #         idx_list = np.argsort(m_abs)[:k-1]
    #         for i in range(k-1):
    #             if m[idx_list[i]] >0.5:
    #                 # set the \hat{x}_{t,(k-1)} to 0
    #                 m_list.append(1*(m - m[idx_list[i]] > 0))
    #             else:
    #                 # set the \hat{x}_{t,(k-1)} to 1
    #                 m_list.append(1*(m - m[idx_list[i]] >= 0))

    #     return m_list

    
    def opn(self, m, k= 1):
        return self.knm(m,k)+self.knm(m+np.random.normal(0,1,len(m)),k)
    
    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]
    
    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))
        
        if k > 1:
            for i in range(k-1):
                m_list.append( np.int64(rng.uniform(low=0.0, high=1.0, size=no_users) < m) ) 

        return m_list
        

    def plot_cost(self, path_name):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.plot(np.arange(len(self.val_cost_his))*self.training_interval, self.val_cost_his)
        plt.ylabel('Training Cost')
        plt.legend(['Train Cost', 'Validation Cost'])
        plt.grid()
        # plt.ylim((0, 1))
        plt.xlabel('Time Frames')
        plt.savefig(path_name)
        plt.show()
