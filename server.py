import numpy as np 
from user import Queue
from system_params import * 
class Server: 
    def __init__(self):
        self.L_i = [Queue(no_slots) for _ in range(no_users)]
        self.b_i = np.zeros((no_slots, no_users))
        self.c_i = np.zeros((no_slots, no_users))
    
    def update_off_task(self, islot, b_i): 
        self.b_i[islot, :] = b_i.reshape(1, -1)
    
    def update_computation_task(self, islot, c_i): 
        self.c_i[islot, :] = c_i

    def update_queue(self, islot):
        for iuser, li in enumerate(self.L_i): 
            li.update(islot=islot, depature=self.c_i[islot-1, iuser], arrival=self.b_i[islot-1, iuser]) 

    def get_queue(self, islot): 
        L_t = np.zeros(no_users)

        for (iuser, li) in enumerate(self.L_i): 
            L_t[iuser] = li.get_queue(islot=islot)
                    
        return L_t 
