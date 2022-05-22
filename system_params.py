
from utils import *


####
# SYSTEM PARAMETERS  
####
no_users = 10
duration = 50 
delta = mini(10) 
no_slots = int(duration/delta)

R = Kb(100)
Amean = Mb(0.1)/R
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
BW_W = MHz(100)


################
## Lyapunov params 
################
LYA_V = 1e7 # Lyapunov


################
## Neural networks 
################
DECODE_MODE = 'OPN'
Memory = 1024          # capacity of memory structure
Delta = 32             # Update interval for adaptive K
no_nn_inputs = 2

############
# Delay 
############
D_TH = 15*Amean


############
# Computation parameters 
#############
KAPPA = 1e-27
fi_0 = giga(0.5)
fu_0 = giga(5)

PSI = 0.001

F = 500*R

pi_0 = dBm(20)

# scale_delay = 10
# d_th = 30

# the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
# decoder_mode = 'OPN'
# Memory = 1024          # capacity of memory structure
# Delta = 32             # Update interval for adaptive K

# CHFACT = 1e12     # The factor for scaling channel value
# QFACT = 1/150     # The factor for scaling channel value
# LFACT = 1/200    # The factor for scaling channel value
# DFACT = 1/3     # The factor for scaling channel value

# mode = 'test'
# # mode = 'ntest'
# window_size = 10
# opt_mode_arr = ['LYDROO', 'bf']
# opt_mode = opt_mode_arr[0]

# no_nn_inputs = 4

# comparison_flag = False 
# V_PHY = 1/Lyapunov_Scale
