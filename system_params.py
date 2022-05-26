
from utils import *


####
# SYSTEM PARAMETERS  
####
no_users = 10
duration = 10 
ts_duration = mini(10) 
no_slots = int(duration/ts_duration)

R = Kb(1)
Amean = 8
Ameans = np.concatenate((Amean*np.ones((no_users, 1))), axis = 0)

#############
## CHANNEL MODEL PARAMETERS 
#############

h_uav = m(50)
a_LOS = 9.16
b_LOS = 0.16
g0 = -50                # channel gain reference dB 
xi = 0.2                # attenuation effect 
mu_gain = 0             # dB fading channel power gain
var_gain = 4            # fading channel variance
sigma_gain = 2          # sqrt(var_gain)
gamma = 2.7601          # path loss exponent 

N0 = dBm(-174)
BW_W = 0.15*no_users*mega(1)


################
## Lyapunov params 
################
LYA_V = 1e5 # Lyapunov


################
## Neural networks 
################
DECODE_MODE = 'OPN'
Memory = 1024          # capacity of memory structure
Delta = 32             # Update interval for adaptive K
no_nn_inputs = 4

############
# Delay 
############
D_TH = 10*Amean


############
# Computation parameters 
#############
KAPPA = 1e-27
fi_0 = giga(0.5)
fu_0 = giga(1.5)
f_iU_0 = giga(7)

PSI = 0.1

F = 500*R

pi_0 = dBm(20)

opt_mode_arr = ['LYDROO', 'bf']
opt_mode = opt_mode_arr[0]