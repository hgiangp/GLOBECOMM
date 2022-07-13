import numpy as np 
from system_params import * 

class TaskChannelGenerator:
    def __init__(self, Ameans):
        self.channel = gen_channel()
        self.task = gen_task(arrival_rates=Ameans)

    def get_channel(self):
        return self.channel

    def get_task(self):
        return self.task

def gen_arrival_task_user(Amean):
    # dataA = np.round(np.random.uniform(0, Amean*2, size=(no_slots, 1)))
    scale_amean = 1 # Amean arrival rate in 
    dataA = np.round(np.random.uniform(0, 2 * Amean/scale_amean, size=(no_slots, 1)))*scale_amean
    return dataA

def gen_gain_slot(dist): 
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

def gen_channel(): 
    '''
    generate channel matrix 
    ''' 
    channel = np.zeros((no_slots, no_users))
    for iuser in range(no_users): 
        positions = np.random.randint(low=10, high=100, size=(no_slots, 1))
        channel[:, iuser] = np.array([gen_gain_slot(dist) for dist in positions]).flatten()
    return channel 

def gen_task(arrival_rates): 
    '''
    generate task 
    ''' 
    task = np.zeros((no_slots, no_users))
    for iuser in range(no_users): 
        task[:, iuser] = gen_arrival_task_user(arrival_rates[iuser]).flatten()
    return task

if __name__ == '__main__': 
    tcg = TaskChannelGenerator(Ameans)
    CHANNEL_FILE = f'./datagen/a{no_users}_{Amean}_channel.npy'
    np.save(CHANNEL_FILE, tcg.get_channel())
    TASK_FILE = f'./datagen/a{no_users}_{Amean}_task.npy'
    np.save(TASK_FILE, tcg.get_task())
    print(f'channel saved to {CHANNEL_FILE}')
    print(f'task saved to {TASK_FILE}')

    # channel = np.load(CHANNEL_FILE)
    # task = np.load(TASK_FILE)

    


