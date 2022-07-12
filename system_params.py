from utils import *

####
# SYSTEM PARAMETERS  
####
only_ue = False   
only_uav = False 

no_users = 10
duration = 100
ts_duration = mini(10) 
no_slots = int(duration/ts_duration)

R = Kb(1)
Amean = 12
Ameans = np.ones(no_users)*Amean

#############
## CHANNEL MODEL PARAMETERS 
#############

h_uav = m(150)
a_LOS = 9.16
b_LOS = 0.16
g0 = -50               # channel gain reference dB 
xi = 0.2                # attenuation effect 
mu_gain = 0             # dB fading channel power gain
var_gain = 4            # fading channel variance
sigma_gain = 2          # sqrt(var_gain)
gamma = 2.7601          # path loss exponent 

N0 = dBm(-174)
if only_ue: 
    BW_W = 1e-7
else:
    BW_W = 0.8*mega(1)


################
## Lyapunov params 
################
LYA_V = 1E6 # Lyapunov
no_cores = 10 

################
## Neural networks 
################
DECODE_MODE = 'OPN'
Memory = 1024          # capacity of memory structure
Delta = 32             # Update interval for adaptive K
no_nn_inputs = 4
learning_rate = 0.001

###########
# Delay 
###########
D_TH = 5 * Amean
# D_TH_arr = np.concatenate((6*np.ones(5), 4*np.ones(no_users - 5)), axis=0)*Amean
D_TH_arr = 5 * Ameans

############
# Computation parameters 
#############
KAPPA = 1e-26
if only_uav: 
    fi_0 = 0
else: 
    fi_0 = giga(0.5)

f_iU_0 = giga(0.8)*10

PSI = 0.1

F = 500*R

pi_0 = dBm(20)

opt_mode_arr = ['learning', 'exhausted', 'random', 'greedy']
opt_mode = opt_mode_arr[1]

########
# pickle file 
########

USERS_FILE = "users.pickle"
SERVER_FILE = "server.pickle"
OPTIMIZER_FILE = "optimizer.pickle"

##########
# DELAY 
##########
delta_t = 2 * Amean
scale_queue = 2 # scaling factor for queue length
