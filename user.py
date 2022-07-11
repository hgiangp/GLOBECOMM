import numpy as np
import matplotlib.pyplot as plt 

from system_params import * 


class Queue: 
    def __init__(self, no_slots):
        self.queue_length = no_slots
        self.value = np.zeros((no_slots, 1))
    
    def update(self, islot, depature, arrival): 
        # self.value[islot, :] = np.maximum(self.value[islot-1, :] - depature + arrival, 0) 
        self.value[islot, :] = np.maximum(self.value[islot-1, :] - depature, 0) + arrival
        pass 
    
    def get_queue(self, islot): 
        return self.value[islot, 0]
class User: 
    def __init__(self, Amean):
        self.gain = dB(self.gen_gain())
        self.A_i = self.gen_arrival_task(Amean)

        self.Q_i = Queue(no_slots=no_slots)

        self._a_i = np.zeros((no_slots, 1))
        self._b_i = np.zeros((no_slots, 1))
        self._delay = np.zeros((no_slots, 1))
        self._pro_energy = np.zeros((no_slots, 1))
        self._off_energy = np.zeros((no_slots, 1))
        self._weighted_energy = np.zeros((no_slots, 1))
        self._virtual_queue = np.zeros((no_slots, 1))
        self.drift = np.zeros((no_slots, 1)) # drift of the user
        self._ts_counter = 0

    def ts_counter(self, value): 
        self._ts_counter = value
    
    def a_i(self, value): 
        self._a_i[self._ts_counter, :] = value

    def b_i(self, value): 
        self._b_i[self._ts_counter, :] = value

    def virtual_queue_i(self, value): 
        self._virtual_queue[self._ts_counter, :] = value

    def pro_energy(self, value): 
        self._pro_energy[self._ts_counter, :] = value

    def off_energy(self, value): 
        self._off_energy[self._ts_counter, :] = value   
   
    def weighted_energy(self, value): 
        self._weighted_energy[self._ts_counter, :] = value

    def delay(self, value): 
        self._delay[self._ts_counter, :] = value 

    def update_queue(self):
        islot = self._ts_counter
        arrival=self.A_i[islot - 1, :]
        departure = self._a_i[islot-1, :] +  self._b_i[islot-1, :]
        self.Q_i.update(islot=islot, depature=departure, arrival= arrival)

    def get_queue(self): 
        return self.Q_i.get_queue(self._ts_counter)

    def gen_arrival_task(self, Amean):
        # dataA = np.round(np.random.uniform(0, Amean*2, size=(no_slots, 1)))
        scale_amean = 1 # Amean arrival rate in 
        dataA = np.round(np.random.uniform(0, 2 * Amean/scale_amean, size=(no_slots, 1)))*scale_amean
        return dataA

    def update_drift(self, drift): 
        self.drift[self._ts_counter, :] = drift
        pass


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