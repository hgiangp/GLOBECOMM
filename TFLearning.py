import numpy as np 
import pickle
import pandas as pd 
import time 
import os 
from pandas import DataFrame as df

from memoryTF2conv import *
from test import plot_optimizer_offloading_decision
from user import User
from server import Server
from system_params import *  
from utils_optimization import *

##### initialization ########

def create_img_folder(): 

    path = "{}/img/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}, no_user={}, delta_t={}/".format(
        os.getcwd(), opt_mode, LYA_V, PSI, D_TH/Amean, Amean, no_users, delta_t)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

pth_folder = create_img_folder()

class TFLearning: 
    def __init__(self):
        self.users = [User(Amean) for Amean in Ameans]
        self.server = Server()
        self.virtualD = np.zeros((no_slots, no_users))
        self.k = no_users       # no of generated modes in each TS 
        self.mem = MemoryDNN(net = [no_users*no_nn_inputs, 256, 128, no_users],
                learning_rate = learning_rate,
                training_interval=20,
                batch_size=128,
                memory_size=Memory)

        self.E_ue_pro = np.zeros((no_slots, no_users)) # energy consumption of all user  
        self.E_ue_off = np.zeros((no_slots, no_users)) # energy consumption of all user  
        self.E_ue = np.zeros((no_slots, no_users))
        # self.E_uav = np.zeros((no_slots, no_users)) # energy consumption of uav for all user 
        self.E_uav = np.zeros((no_slots, 1))
        self.delay = np.zeros((no_slots, no_users))
        self.bf_action = gen_actions_bf(no_users=no_users)
        self.mode_his = np.zeros((no_slots, no_users))
        self.Mt_s = np.zeros((no_slots))

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
            user.a_i(a_i[iuser])
            user.b_i(b_i[iuser])
            user.weighted_energy(self.E_ue[islot, iuser])
            user.pro_energy(self.E_ue_pro[islot, iuser])
            user.off_energy(self.E_ue_off[islot, iuser])
            user.delay(self.delay[islot, iuser])
            user.virtual_queue_i(self.virtualD[islot, iuser])
            # update tasks 

        # update server queue 
        self.server.b_i(b_i=b_i)
        self.server.c_i(c_i=c_i)
        self.server.energy(self.E_uav[islot, :])
        self.server.virtual_queue(self.virtualD[islot, :])

    def update_n_get_queue(self, islot): 
        for user in self.users: 
            user.update_queue()
        self.server.update_queue()

        Q_t = np.array([user.get_queue() for user in self.users])
        L_t = self.server.get_queue()
        self.virtualD[islot, :] = np.maximum(self.virtualD[islot-1, :] + (Q_t + L_t + self.mode_his[islot-1, :]*delta_t) - D_TH_arr, 0)
        
        self.update_drift(islot) # update drift

        return Q_t, L_t, self.virtualD[islot, :]
    
    def update_drift(self, islot):
        drift = 1/2 * (self.virtualD[islot, :] - self.virtualD[islot-1, :])** 2 
        for (id, user) in enumerate(self.users): 
            user.update_drift(drift[id])
    
    def get_queue(self, islot): 
        return self.virtualD[islot, :]

    def cal_delay2(self, islot):
        Q_t = np.concatenate([user.Q_i.value[:islot+1, :] for user in self.users], axis=1) 
        L_t = np.concatenate([li.value[:islot+1, :] for li in self.server.L_i], axis=1)
        time_delta_t = np.mean(self.mode_his[:islot-1, :], axis=0)*delta_t if islot > 1 else 0 
        delay = (np.mean(Q_t + L_t, axis=0) + time_delta_t)/Amean
        return delay

    def cal_delay(self, islot):
        window_size = 30
        end_index = islot + 1  
        start_index = end_index - window_size + 1 
        Q_t = np.concatenate([user.Q_i.value[start_index:end_index, :] for user in self.users], axis=1) 
        L_t = np.concatenate([li.value[start_index:end_index, :] for li in self.server.L_i], axis=1)
        time_delta_t = np.mean(self.mode_his[start_index:end_index-1, :], axis=0)*delta_t
        delay = (np.mean(Q_t + L_t, axis=0) + time_delta_t)/Amean
        return delay

    def learning(self): 
        k_idx_his = []
        for islot in range(0, no_slots): 
            if islot % (no_slots//10) == 0:
                print("%0.1f"%(islot/no_slots))
            if islot > 0 and islot % Delta == 0:
            # index counts from 0
                if Delta > 1:
                    max_k = max(np.array(k_idx_his[-Delta:-1])%self.k) +1
                else:
                    max_k = k_idx_his[-1] +1
                self.k = min(max_k +1, no_users)
            #######upate M_t ################
            self.Mt_s[islot] = self.k 
            ################################
            ## update counter
            for user in self.users: 
                user.ts_counter(islot)
            self.server.ts_counter(islot)
            
            # update queue 
            gain_t = self.get_gain_ue(islot=islot)
            Q_t, L_t, D_t = self.update_n_get_queue(islot=islot)


            # normalize and get nn input  
            h_norm, d_norm = self.norm_input_nn(islot=islot)
            q_norm, l_norm = preprocessing(Q_t), preprocessing(L_t)

            nn_input = np.vstack((h_norm, q_norm, l_norm, d_norm)).transpose().flatten()
            if opt_mode == opt_mode_arr[1]: 
                m_list = self.bf_action.copy()
            elif opt_mode == opt_mode_arr[0]: 
                m_list = self.mem.decode(nn_input, self.k, DECODE_MODE)
            elif opt_mode == opt_mode_arr[2]: 
                no_bf_actions = self.bf_action.shape[0]
                random_index = np.random.choice(no_bf_actions, 1)
                m_list = self.bf_action[random_index, :]
            elif opt_mode == opt_mode_arr[3]:
                m_list = gen_actions_greedy_queue(D_t)
            
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
            self.mode_his[islot, :] = m_list[best_idx]


            # 3. policy update module
            # encode with the larget reward 
            self.mem.encode(nn_input, m_list[best_idx])
            
            tmp, a_t, b_t, c_t, self.E_ue_pro[islot, :], self.E_ue_off[islot, :], self.E_uav[islot, :] = r_list[best_idx]
            self.E_ue[islot, :] = self.E_ue_pro[islot, :] + self.E_ue_off[islot, :]
            # calculate delay 
            self.delay[islot, :] = self.cal_delay2(islot)

            self.return_opt_value(islot=islot, a_i=a_t, b_i=b_t, c_i=c_t)

            is_debug_mode = True

            if is_debug_mode and islot%100 == 0: 
                print(f'local computation: a_i =', a_t)
                print(f'offloading volume: b_i =', b_t)
                print(f'remote computation: c_i =', c_t)
                print(f'remote computation: energy_i =', self.E_ue_pro[islot, :]*1000)
                print(f'remote computation: energy_i =', self.E_ue_off[islot, :]*1000)
                print(f'remote computation: energy_u =', self.E_uav[islot, :]/no_users*1000)
                print(f'virtual queue_i =', self.virtualD[islot,:])                
                print(f'fvalue = {v_list[k_idx_his[-1]]}')
        
        
        save_data(file_name = pth_folder + USERS_FILE, object=self.users)
        save_data(file_name = pth_folder + SERVER_FILE, object=self.server)
        
        print("finished!")

    def plot_figure(self, running_time, pth_folder): 
        

        self.mem.plot_cost(pth_folder + "Training Loss")        
        
        Q_i = np.concatenate([user.Q_i.value[:, :] for user in self.users], axis=1)
        Q_mean = np.mean(Q_i, axis=1)
        L_mean = np.mean(np.concatenate([li.value for li in self.server.L_i], axis = 1), axis=1)
        D_mean = np.mean(self.virtualD, axis=1)
        E_ue_pro_mean = np.mean(self.E_ue_pro, axis=1)*1000/ts_duration
        E_ue_off_mean = np.mean(self.E_ue_off, axis=1)*1000/ts_duration
        E_ue_mean = E_ue_pro_mean + E_ue_off_mean
        E_uav_mean = np.mean(self.E_uav, axis=1)*1000/ts_duration
        W_E_mean = E_ue_mean + PSI * E_uav_mean
        ava_delay = np.mean(self.delay, axis=1)
    
        a_mean = np.mean(np.concatenate([user._a_i for user in self.users], axis=1), axis=1) 
        b_mean = np.mean(np.concatenate([user._b_i for user in self.users], axis=1), axis=1) 
        c_mean = np.mean(self.server._c_i, axis=1)

        df = pd.DataFrame( {'local_queue':Q_mean,'uav_queue':L_mean,
            'energy_user_pro':E_ue_pro_mean,'energy_user_off':E_ue_off_mean, 
            'energy_user':E_ue_mean,'energy_uav':E_uav_mean, 
            'delay':ava_delay, 'weightedE': W_E_mean,  
            'off_b': b_mean, 'local_a': a_mean, 'remote_c': c_mean, 
            'time': running_time
            })
        df.to_csv(pth_folder+"result.csv",index=False)

def save_data(file_name, object):
    with open(file_name, 'wb') as handle: 
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__": 
    
    optimizer = TFLearning()
    start_time = time.time()
    optimizer.learning()
    total_time = time.time() - start_time
    print("{:.2f}".format(total_time))

    path = create_img_folder()
    optimizer.plot_figure(running_time=total_time, pth_folder=path)
    # save_data(file_name = pth_folder + OPTIMIZER_FILE, object=optimizer)
    plot_optimizer_offloading_decision(optimizer, path)
    # import matplotlib.pyplot as plt
    # plt.plot(optimizer.Mt_s)
    # plt.show()
    ### plot number of offloading users
    
    print('finish')
