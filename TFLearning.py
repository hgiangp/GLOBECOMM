import numpy as np 
import pickle
import pandas as pd 
import time 
from pandas import DataFrame as df

from memoryTF2conv import *
from plot_kpi import create_img_folder, plot_kpi_avr, plot_kpi_users, plot_offloading_computation, plot_pickle, plot_rate, save_data 
from user import User
from server import Server
from system_params import *  
from utils_optimization import * 

##### initialization ########

class TFLearning: 
    def __init__(self):
        self.users = [User(Amean) for Amean in Ameans]
        self.server = Server()
        self.virtualD = np.zeros((no_slots, no_users))
        self.k = 100         # no of generated modes in each TS 
        self.mem = MemoryDNN(net = [no_users*no_nn_inputs, 256, 128, no_users],
                learning_rate = 0.01,
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
            user.delay(self.delay[islot, iuser])
            # update tasks 

        # update server queue 
        self.server.b_i(b_i=b_i)
        self.server.c_i(c_i=c_i)
        self.server.energy(self.E_uav[islot, :])

    def update_n_get_queue(self, islot): 
        for user in self.users: 
            user.update_queue()
        self.server.update_queue()

        Q_t = np.array([user.get_queue() for user in self.users])
        L_t = self.server.get_queue()    
        self.virtualD[islot, :] = np.maximum(self.virtualD[islot-1, :] + (Q_t + L_t) - D_TH_arr, 0) 

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
        for islot in range(0, no_slots): 
            if islot % (no_slots//10) == 0:
                print("%0.1f"%(islot/no_slots))
            
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
            if opt_mode == 'bf': 
                m_list = self.bf_action.copy()
            elif opt_mode == 'lydroo': 
                m_list = self.mem.decode(nn_input, self.k, DECODE_MODE)
            elif opt_mode == 'random': 
                random_index = np.random.choice(2**no_users, 1)
                m_list = self.bf_action[random_index, :]
            
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
            
            tmp, a_t, b_t, c_t, self.E_ue_pro[islot, :], self.E_ue_off[islot, :], self.E_uav[islot, :] = r_list[best_idx]
            self.E_ue[islot, :] = self.E_ue_pro[islot, :] + self.E_ue_off[islot, :]
            # calculate delay 
            self.delay[islot, :] = self.cal_delay(islot)

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
        
        pth_folder = create_img_folder()
        save_data(file_name = pth_folder + "users.pickle", object=self.users)
        save_data(file_name = pth_folder + "server.pickle", object=self.server)
        
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
        E_uav_mean = np.mean(self.E_uav, axis=1)/no_users*1000/ts_duration
        W_E_mean = E_ue_mean + PSI * E_uav_mean
        ava_delay = np.mean(self.delay, axis=1)


        kpi_list_user =  ['Average computation power', 'Average offloading power', 'Total power']
        data_list_user = [E_ue_pro_mean, E_ue_off_mean, E_ue_mean]
        plot_kpi_users(data_list_user, kpi_list_user, path = pth_folder, title ='users')

        a_mean = np.mean(np.concatenate([user._a_i for user in self.users], axis=1), axis=1) 
        b_mean = np.mean(np.concatenate([user._b_i for user in self.users], axis=1), axis=1) 
        c_mean = np.mean(self.server._c_i, axis=1)
        plot_offloading_computation(b_mean, c_mean, pth_folder)

        kpi_list = ['User Queue Length (packets)', 'UAV queue length (packets)', 'Virtual queue', 'User power (mW)', 'UAV power (mW)', 'Weighted power', 'Delay']
        data_list = [Q_mean, L_mean, D_mean, E_ue_mean, E_uav_mean, W_E_mean, ava_delay]

        plot_kpi_avr(data_list, kpi_list, path=pth_folder)

        df = pd.DataFrame( {'local_queue':Q_mean,'uav_queue':L_mean,
            'energy_user_pro':E_ue_pro_mean,'energy_user_off':E_ue_off_mean, 
            'energy_user':E_ue_mean,'energy_uav':E_uav_mean, 
            'delay':ava_delay, 'weightedE': W_E_mean, 
            'off_b': b_mean, 'local_a': a_mean, 'remote_c': c_mean, 
            'time': running_time
            })
        df.to_csv(pth_folder+"result.csv",index=False)
        

if __name__ == "__main__": 
    
    optimizer = TFLearning()
    start_time = time.time()
    optimizer.learning()
    total_time = time.time() - start_time
    print("{:.2f}".format(total_time))

    path = create_img_folder()
    
    optimizer.plot_figure(running_time=total_time, pth_folder=path)
    plot_pickle(path)


    print('finish')
