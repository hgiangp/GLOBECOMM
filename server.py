import numpy as np 
from user import Queue
from system_params import * 
class Server: 
    def __init__(self):
        self.L_i = [Queue(no_slots) for _ in range(no_users)]
        self._b_i = np.zeros((no_slots, no_users))
        self._c_i = np.zeros((no_slots, no_users))
        self._energy = np.zeros((no_slots, no_users))
        self._ts_counter = 0 

    def ts_counter(self, value): 
        self._ts_counter = value 
    
    def b_i(self, b_i): 
        self._b_i[self._ts_counter, :] = b_i.reshape(1, -1)
    
    def c_i(self, c_i): 
        self._c_i[self._ts_counter, :] = c_i
    
    def energy(self, energy_users): 
        self._energy[self._ts_counter, :] = energy_users 

    def update_queue(self):
        islot = self._ts_counter
        for iuser, li in enumerate(self.L_i): 
            li.update(islot=islot, depature=self._c_i[islot-1, iuser], arrival=self._b_i[islot-1, iuser]) 

    def get_queue(self):
        islot = self._ts_counter 
        L_t = np.zeros(no_users)

        for (iuser, li) in enumerate(self.L_i): 
            L_t[iuser] = li.get_queue(islot=islot)
                    
        return L_t 
