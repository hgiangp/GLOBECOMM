from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt 

from system_params import * 

class User: 
    def __init__(self, Amean): 
        self.a_i = np.zeros((no_slots))
        self.b_i = np.zeros((no_slots))
        self.Q_i = np.zeros((no_slots))
        self.A_i = self.gen_arrival_task(Amean)
        self.gain = dB(self.gen_gain())
        self.E_i = np.zeros((no_slots))


    def update_phy_queue(self, islot, a_i, b_i):
        self.a_i[islot-1] = a_i
        self.b_i[islot-1] = b_i 

        self.Q_i[islot] = max(self.Q_i[islot-1] - a_i - b_i, 0) + self.A_i[islot-1]
        
        return self.Q_i[islot]
    
    def update_energy(self, islot, E_t): 
        # a_i = f*delta/F -> f = a_i*F/delta 
        # e_ai = kappa * (a_i*F/delta)**3 ) * delta 
        # b_i = bw*delta/R * log2 (1 + ph/(N0*bw))
        # e_bi = (N0*bw/h) * exp((b_i*R*ln(2)/(bw*delta)) - 1)
        self.E_i[islot] = E_t 

        

    def gen_arrival_task(self, Amean):
        dataA = np.random.uniform(0, Amean*2, size=(no_slots, 1))
        return dataA 


    def gen_gain_slot(self, dist): 
        '''
        generate channel gain in dB of 1 user 
        ''' 

        theta = np.arctan(h_uav/dist)
        fading = np.random.normal(mu_gain, sigma_gain)

        p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

        # assume we omit the small scale fading 
        gain_no_fading = todB(p_LOS + xi*(1 - p_LOS)) + g0 - todB((h_uav**2 + dist**2)**(gamma/2))
        gain_fading = gain_no_fading + fading 

        return gain_fading

    def gen_gain(self): 
        dist_arr = np.random.randint(low=10, high=100, size=(no_slots, 1))
        gain = np.array([self.gen_gain_slot(dist) for dist in dist_arr])
        return gain 
    

# if name == '__main__': 
A = User(200000)
print('finished')
