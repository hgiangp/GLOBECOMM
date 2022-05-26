import numpy as np 
import pickle
import pandas as pd 
import time 
from pandas import DataFrame as df

from memoryTF2conv import *
from plot_kpi import create_img_folder, plot_kpi_avr, plot_kpi_users, plot_rate 
from user import User
from server import Server
from system_params import *  

import utils
from utils_optimization import * 

##### initialization ########

class TFLearning: 
    def __init__(self):
        self.users = [User(Amean) for Amean in Ameans]
        self.server = Server()
        self.virtualD = np.zeros((no_slots, no_users))
        self.k = 100           # no of generated modes in each TS 
        self.mem = MemoryDNN(net = [no_users*no_nn_inputs, 256, 128, no_users],
                learning_rate = 0.01,
                training_interval=20,
                batch_size=128,
                memory_size=Memory)

        self.E_ue = np.zeros((no_slots, no_users)) # energy consumption of all user  
        self.E_uav = np.zeros((no_slots, no_users)) # energy consumption of uav for all user 
        self.delay = np.zeros((no_slots, no_users))

    def get_gain_ue(self, islot): 
        gain = np.array([user.gain[islot] for user in self.users])
        gain = gain.reshape(-1)
        return gain

    def norm_input_nn(self, islot):
        # get nn inputs  
        gain = np.array([user.gain[islot] for user in self.users]) # channel gain 
        D_t = self.get_queue(islot=islot)
        # normalize the input 
        gain_norm = preprocessing(gain)
        dt_norm = preprocessing(D_t)
        return gain_norm, dt_norm 


    def return_opt_value(self, islot, a_i, b_i, c_i):
        # update users queue
        for (iuser, user) in enumerate(self.users): 
            # update tasks of this time slot 
            user.update_computation_task(islot, a_i[iuser])
            user.update_offload_task(islot, b_i[iuser])
            # update tasks 

        # update server queue 
        self.server.update_off_task(islot=islot, b_i=b_i)
        self.server.update_computation_task(islot=islot, c_i=c_i)

    def update_n_get_queue(self, islot): 
        for user in self.users: 
            user.update_queue(islot=islot)
        self.server.update_queue(islot=islot)

        Q_t = np.array([user.get_queue(islot=islot) for user in self.users])
        L_t = self.server.get_queue(islot=islot)    
        self.virtualD[islot, :] = np.maximum(self.virtualD[islot-1, :] + (Q_t + L_t) - D_TH, 0) 

        return Q_t, L_t, self.virtualD[islot, :]
    
    def get_queue(self, islot): 
        return self.virtualD[islot, :]

    def cal_delay(self, islot):
        Q_t = np.concatenate([user.Q_i.value[:islot+1, :] for user in self.users], axis=1) 
        L_t = np.concatenate([li.value[:islot+1, :] for li in self.server.L_i], axis=1)
        
        delay = np.mean(Q_t + L_t, axis=0)/Amean
        return delay 

    def learning(self): 
        k_idx_his = []
        for islot in range(1, no_slots): 
            if islot % (no_slots//10) == 0:
                print("%0.1f"%(islot/no_slots))
            
            # update queue 
            gain_t = self.get_gain_ue(islot=islot)
            Q_t, L_t, D_t = self.update_n_get_queue(islot=islot)

            # normalize and get nn input  
            h_norm, d_norm = self.norm_input_nn(islot=islot)
            q_norm, l_norm = preprocessing(Q_t), preprocessing(L_t)

            nn_input = np.vstack((h_norm, q_norm, l_norm, d_norm)).transpose().flatten()

            m_list = self.mem.decode(nn_input, self.k, DECODE_MODE)
            
            r_list = []
            v_list = []
            
            for m in m_list: 
                # 2. critic module of LyDROO
                # allocate resource for all offloading mode 
                r_list.append(resource_allocation(off_decsion=m, Q=Q_t, L=L_t, gain=gain_t, D = D_t))
                v_list.append(r_list[-1][0])
            # record the largest reward 
            best_idx = np.argmin(v_list)
            k_idx_his.append(np.argmin(v_list))


            # 3. policy update module
            # encode with the larget reward 
            self.mem.encode(nn_input, m_list[best_idx])
                        
            # store max result 
            tmp, a_t, b_t, c_t, self.E_ue[islot, :], self.E_uav[islot, :] = r_list[best_idx]

            self.return_opt_value(islot=islot, a_i=a_t, b_i=b_t, c_i=c_t)

            # calculate delay 
            self.delay[islot, :] = self.cal_delay(islot)

            is_debug_mode = True

            if is_debug_mode and islot%1 == 0: 
                print(f'local computation: a_i =', a_t)
                print(f'offloading volume: b_i =', b_t)
                print(f'remote computation: c_i =', c_t)
                print(f'remote computation: energy_i =', self.E_ue[islot, :]*1000)
                print(f'remote computation: energy_u =', self.E_uav[islot, :]*1000)
                print(f'virtual queue_i =', self.virtualD[islot,:])                
                print(f'fvalue = {v_list[k_idx_his[-1]]}')

    print("finished!")

    def plot_figure(self): 
        pth_folder = create_img_folder()

        self.mem.plot_cost(pth_folder + "Training Loss")

        kpi_list = ['Arrival tasks', 'Local tasks', 'Offloaded tasks', 'Queue length']
        for (iuser, user) in enumerate(self.users): 
            data_list_ue = [user.A_i, user.a_i.reshape(-1), user.b_i.reshape(-1), user.Q_i.value.reshape(-1)]
            plot_kpi_users(data_list_ue, kpi_list, path = pth_folder, title ='user[{}]'.format(iuser))
        
        kpi_list = ['Arrival task', 'Computation tasks', 'Queue length']
        Q_i = np.concatenate([user.Q_i.value[:, :] for user in self.users], axis=1)
        ava_server_queue = np.mean(np.concatenate([li.value for li in self.server.L_i], axis = 1), axis=1)
        data_list_uav = [np.mean(self.server.b_i, axis=1), np.mean(self.server.c_i, axis = 1), ava_server_queue]
        plot_kpi_users(data_list_uav, kpi_list, path = pth_folder, title ='server')

        
        Q_mean = np.mean(Q_i, axis=1)
        L_mean = ava_server_queue.copy()
        D_mean = np.mean(self.virtualD, axis=1)
        E_ue_mean = np.mean(self.E_ue, axis=1)*1000/ts_duration
        E_uav_mean = np.mean(self.E_uav, axis=1)*1000/ts_duration
        W_E_mean = E_ue_mean + PSI * E_uav_mean
        ava_delay = np.mean(self.delay, axis=1)

        a_mean = np.mean(np.concatenate([user.a_i for user in self.users], axis=1), axis=1) 
        b_mean = np.mean(np.concatenate([user.b_i for user in self.users], axis=1), axis=1) 
        c_mean = np.mean(self.server.c_i, axis=1)

        kpi_list = ['User Queue Length (packets)', 'UAV queue length (packets)', 'Virtual queue', 'User power (mW)', 'UAV power (mW)', 'Weighted power', 'Delay']
        data_list = [Q_mean, L_mean, D_mean, E_ue_mean, E_uav_mean, W_E_mean, ava_delay]

        plot_kpi_avr(data_list, kpi_list, path=pth_folder)

        df = pd.DataFrame( {'local_queue':Q_mean,'uav_queue':L_mean,
            'energy_user':E_ue_mean,'energy_uav':E_uav_mean, 
            'delay':ava_delay, 'weightedE': W_E_mean, 
            'off_b': b_mean, 'local_a': a_mean, 'remote_c': c_mean, 
            # 'time': total_time
            })
        df.to_csv(pth_folder+"result.csv",index=False)

if __name__ == "__main__": 
    
    optimizer = TFLearning()
    optimizer.learning()
    optimizer.plot_figure()



    

    

    # object = optimizer
    # filename = "users.pickle"
    # filehandler = open(filename, 'wb') 
    # pickle.dump(object, filehandler)
    # filehandler.close()

    print('finish')
