from tkinter import N
from memoryTF2conv import MemoryDNN
from server import Server 
from user import User

from system_params import * 

class CServer: 
    def __init__(self):
        self.uav = Server()
        self.users = [User(Amean) for Amean in Ameans]
        self.D = np.zeros((no_slots, no_users))
        self.k = 100           # no of generated modes in each TS 
        self.obj_value = np.zeros((no_slots)) 
        self.mem = MemoryDNN(net = [no_users*no_nn_inputs, 256, 128, no_users],
                learning_rate = 0.01,
                training_interval=20,
                batch_size=128,
                memory_size=Memory)

    def cal_obj_func(func, off_decision, a_t, b_t, c_t, Q_t, L_t, D_t): 
        f_i = a_t*F/delta
        obj_pb1 = KAPPA*(f_i**3)*delta - (Q + )
        f_u = c_i*F/delta 
        
        
    
    def update_queue(self, islot, a_i, b_i, c_i):
        
        # update UE's queue 
        Q_tp1 = np.array([user.update_phy_queue(islot, a_i=a_i[0, iuser], b_i=b_i[0, iuser]) for (iuser, user) in enumerate(self.users)])
        # update UAV's queue
        L_tp1 = self.uav.update_phy_queue(islot, c_i, b_i)
        # update queue
        Q_tot_tp1 = np.add(Q_tp1, L_tp1)
        # update virtual queue 
        self.D[islot, :] = np.maximum(np.add(self.D[islot-1, :], Q_tot_tp1) - D_TH, 0) 
        
        return self.D[islot, :]

    def opt_local_computation(self, off_decision, Q_t, D_t):
        a_opt = np.zeros((no_users)) # optimal local frequency

        idx_loc_ue = off_decision == 0 
        no_loc_ue = no_users - np.sum(off_decision)

        if no_loc_ue != 0:
            f_hat = np.minimum(fi_0, Q_t[idx_loc_ue]*F/delta)
            f_l = np.sqrt((Q_t[idx_loc_ue] + D_t[idx_loc_ue])/(3*LYA_V*KAPPA*PSI*F))
            f_l = np.minimum(f_l, f_hat)

            a_opt[idx_loc_ue] = f_l*delta/F
        
        return a_opt


    def opt_offloading_eqbw(self, off_decision, Q_t, L_t, h_t): 
        b_opt = np.zeros((no_users))
        bw_opt = np.zeros((no_users))
        no_off_ue = np.sum(off_decision)
        bhat = np.zeros((no_users))


        if no_off_ue != 0: 
            idx_off_ue = (off_decision == 1) 
            idx_queue = Q_t > L_t 
            idx_ue = idx_off_ue & idx_queue 

            eqbw = BW_W/no_off_ue
            bw_opt[idx_ue] = eqbw

            bhat[idx_ue] = np.minimum(Q_t[idx_ue], eqbw*delta/R*np.log2(1 + pi_0*h_t[idx_ue]/N0/eqbw))
            b_opt[idx_ue] = np.minimum(bhat[idx_ue], eqbw*delta/R*np.log2((Q_t[idx_ue] - L_t[idx_ue])*h_t[idx_ue]/(LYA_V*N0*R*np.log(2))))
            b_opt[idx_ue] = np.maximum(0, np.round(b_opt[idx_ue]))
            
            return b_opt, bw_opt 

    def opt_uav_computation(self, off_decision, L_t, D_t): 
        c_opt = np.zeros((no_users))

        idx_loc_ue = off_decision == 1 
        no_loc_ue = no_users - np.sum(off_decision)

        if no_loc_ue != 0:
            f_hat = np.minimum(fi_0, L_t[idx_loc_ue]*F/delta)
            f_u = np.sqrt((L_t[idx_loc_ue] + D_t[idx_loc_ue]) / (3*LYA_V*KAPPA*PSI*F))
            f_u = np.minimum(f_u, f_hat)

            c_opt[idx_loc_ue] = f_u*delta/F
            return c_opt 

    def opt_resource_allocation(self, islot, off_decision): 
        # opt problem 1
        gain = np.array([user.gain[islot, 0] for user in self.users])
        Q_t = np.array([user.Q_i[islot] for user in self.users])
        L_t = self.uav.L[islot, :]
        D_t = self.D[islot, :] 
        # get result 
        a_opt = self.opt_local_computation(off_decision=off_decision, Q_t=Q_t, D_t=D_t)
        b_opt, bw_opt = self.opt_offloading_eqbw(off_decision=off_decision, Q_t=Q_t, L_t=L_t, h_t=gain)
        c_opt = self.opt_uav_computation(off_decision=off_decision, L_t=L_t, D_t=D_t)

        return 

    def running(self): 
        k_idx_his = []

        a_t = np.zeros((1, no_users))
        b_t = np.zeros((1, no_users))
        c_t = np.zeros((1, no_users))
        E_ue = np.zeros((no_slots, no_users))
        E_uav = np.zeros((no_slots, no_users))

        for islot in range(1, no_slots): 
            if islot % (no_slots//10) == 0:
                print("%0.1f"%(islot/no_slots))

            # DNN input  
            gain_h = np.array([user.gain[islot] for user in self.users])
            # Update virtual Q 
            delay_queue = self.update_queue(islot=islot, a_i = a_t, b_i = b_t, c_i = c_t)

            # normalize the input of the DNN 
            h_norm = preprocessing(gain_h)
            d_norm = preprocessing(delay_queue)

            nn_input = np.vstack((h_norm, d_norm)).transpose().flatten()

            m_list = self.mem.decode(nn_input, self.k, DECODE_MODE)
            r_list = []
            v_list = []
            for m in m_list: 
                # 2. critic module of LyDROO
                # allocate resource for all offloading mode 
                r_list.append(self.opt_resource_allocation(islot=islot, off_decision=m))
                v_list.append(r_list[-1][0])
            # record the largest reward 
            best_idx = np.argmin(v_list)
            k_idx_his.append(np.argmin(v_list))


            # 3. policy update module
            # encode with the larget reward 
            self.mem.encode(nn_input, m_list[best_idx])
            
            # store max result 
            self.obj_value[islot], a_t, b_t, c_t, E_ue[islot, :], E_uav[islot, :] = r_list[best_idx]

            is_debug_mode = True             

            if is_debug_mode and islot%1 == 0: 
                print(f'local computation: a_i =', a_t)
                print(f'offloading volume: b_i =', b_t)
                print(f'remote computation: c_i =', c_t)
                print(f'remote computation: energy_i =',E_ue[islot, :]*1000)
                print(f'remote computation: energy_u =', E_uav[islot, :]*1000)
                print(f'virtual queue_i =', self.D[islot,:])                
                print(f'fvalue = {v_list[k_idx_his[-1]]}')
        self.mem.plot_cost("Training Loss")

        print("finished!")
        

            



            
            

            

            
         

