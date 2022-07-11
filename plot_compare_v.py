######### LOAD Data #########
from test import load_data_pickle, create_img_folder, plot_moving_average
from system_params import *

import matplotlib.pyplot as plt


LYA_V = 5*1e4
Amean = 7
D_TH = 10 * Amean
no_users = 10
delta_t = 5 * Amean
opt_mode = 'learning'
v_array = [1e5, 1e6]


ue_pro_power_array = np.zeros((len(v_array), no_slots))
ue_off_power_array = np.zeros((len(v_array), no_slots))


for idx, LYA_V in enumerate(v_array): 
    
    path_dir = create_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, PSI=PSI, D_TH=D_TH, Amean=Amean, delta_t=delta_t)
    
    print(path_dir) 

    users = load_data_pickle(file_name=path_dir + USERS_FILE)
    server = load_data_pickle(file_name=path_dir + SERVER_FILE)


    ####################
    # save data 
    ####################

    ue_pro_power_arr = np.zeros((no_slots, no_users))
    ue_off_power_arr = np.zeros((no_slots, no_users))

    for iuser, user in enumerate(users): 
        ue_pro_power_arr[:, iuser] = user._pro_energy[:, 0]/ts_duration*1000 # mW 
        ue_off_power_arr[:, iuser] = user._off_energy[:, 0]/ts_duration*1000 # mW

    #######UAV #########
    ue_pro_power_array[idx, :] = np.mean(ue_pro_power_arr, axis=1).reshape(no_slots)
    ue_off_power_array[idx, :] = np.mean(ue_off_power_arr, axis=1).reshape(no_slots)


    ###############
    # plot
    pass 


fig = plt.figure()
color_list = [['b', 'r'], ['g', 'c']]
rolling_intv = 300 
for iv, lya_v in enumerate(v_array): 
    data = ue_pro_power_array[iv, :]
    plot_moving_average(data, color=color_list[iv][0], label='p_pro V = {:.1e}'.format(lya_v), rolling_intv=rolling_intv)
    data = ue_off_power_array[iv, :]
    plot_moving_average(data, color=color_list[iv][1], label='p_off V = {:.1e}'.format(lya_v), rolling_intv=rolling_intv)

plt.grid()
plt.legend()
plt.ylim(0, 140)
plt.xlim(0, 10000)
plt.ylabel('Average user power (mW)')
import os 
path_dir = os.getcwd() + '/' + 'img/'
file_ext = ['eps', 'png']
plt.savefig(path_dir + 'compare_user_power.eps')
plt.savefig(path_dir + 'compare_user_power.png')
print('Saved: ' + path_dir + 'compare_user_power.eps')
plt.show()

plt.show()


