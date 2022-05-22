import numpy as np 
from system_params import * 


class Server: 
    def __init__(self):
        self.b = np.zeros((no_slots, no_users))
        self.c = np.zeros((no_slots, no_users))
        self.L = np.zeros((no_slots, no_users))
    
    def update_phy_queue(self, islot, c, b): 
        self.b[islot-1, :] = b
        self.c[islot-1, :] = c 

        self.L[islot, :] = np.maximum(self.L[islot-1, :] - c, 0) + b 
        
        return self.L[islot, :]
