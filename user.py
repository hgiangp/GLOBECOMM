import numpy as np
import matplotlib.pyplot as plt 

from system_params import * 


class Queue: 
    def __init__(self, no_slots):
        self.queue_length = no_slots
        self.value = np.zeros((no_slots, 1))
    
    def update(self, islot, depature, arrival): 
        self.value[islot, :] = np.maximum(self.value[-1, :] - depature, 0) + arrival
    
    def get_queue(self, islot): 
        return self.value[islot, 0]
class User: 
    def __init__(self, Amean):
        self.gain = dB(self.gen_gain())
        self.A_i = self.gen_arrival_task(Amean)

        self.Q_i = Queue(no_slots=no_slots)

        self.a_i = np.zeros((no_slots, 1))
        self.b_i = np.zeros((no_slots, 1))
    
    def update_computation_task(self, islot, value): 
        self.a_i[islot, :] = value 

    def update_offload_task(self, islot, value): 
        self.b_i[islot, :] = value  

    def update_queue(self, islot):
        self.Q_i.update(islot=islot, depature=self.a_i[islot-1, :] + \
            self.b_i[islot-1, :], arrival=self.A_i[islot - 1, :])

    def get_queue(self, islot): 
        return self.Q_i.get_queue(islot)

    def gen_arrival_task(self, Amean):
        dataA = np.round(np.random.uniform(0, Amean*2, size=(no_slots, 1)))
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
