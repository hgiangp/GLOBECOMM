######### LOAD Data #########
from test import load_data_pickle, create_img_folder, plot_moving_average
from system_params import *

import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'Helvetica'})

LYA_V = 1e6
Amean = 9
D_TH = 5 * Amean
no_users = 10 
delta_t = 2 * Amean
opt_mode_arr = ['greedy', 'learning', 'exhausted']

weighted_power_arr = np.zeros((no_slots, len(opt_mode_arr)))

for idx, mode in enumerate(opt_mode_arr): 
    print(D_TH)
    path_dir = create_img_folder(opt_mode=mode, LYA_V=LYA_V, PSI=PSI, D_TH=D_TH, Amean=Amean, delta_t=delta_t, no_users=no_users)
    
    print(path_dir) 

    users = load_data_pickle(file_name=path_dir + USERS_FILE)
    server = load_data_pickle(file_name=path_dir + SERVER_FILE)


    ####################
    # save data 
    ####################

    ue_pro_power_arr = np.zeros((no_slots, no_users))
    ue_off_power_arr = np.zeros((no_slots, no_users))
    ue_total_power_arr = np.zeros((no_slots, no_users))
    uav_power_arr = np.zeros((no_slots))

    for iuser, user in enumerate(users): 
        ue_pro_power_arr[:, iuser] = user._pro_energy[:, 0]/ts_duration*1000 # mW 
        ue_off_power_arr[:, iuser] = user._off_energy[:, 0]/ts_duration*1000 # mW
        ue_total_power_arr[:, iuser] = ue_pro_power_arr[:, iuser] + ue_off_power_arr[:, iuser]

    #######UAV #########
    uav_power_arr = server._energy[:, 0]/ts_duration*1000
    ue_power_arr = np.sum(ue_total_power_arr, axis=1)

    weighted_power = ue_power_arr + PSI * uav_power_arr

    weighted_power_arr[:, idx] = weighted_power[:]/no_users # average power of each slot for all users 

    ###############
    # plot
    pass 


fig = plt.figure()
color_list = ['tab:blue', 'tab:green', 'tab:orange']
label_list = ['Max-Queue', 'Learning', 'Exhausted']
rolling_intv = 20
for idata, opt_mode in enumerate(opt_mode_arr): 
    data = weighted_power_arr[:, idata]
    plot_moving_average(data, color=color_list[idata], label=label_list[idata], rolling_intv=rolling_intv)
plt.legend(fontsize=12, loc='lower right')
plt.ylim(200, 1000)
plt.xlim(0, 3000)
plt.grid()
plt.ylabel('Average weighted power (mW)', fontsize=12)
plt.xlabel('Time Frame', fontsize=12)
plt.savefig('./results/' + 'power_vs_time.eps', bbox_inches='tight')
plt.savefig('./results/' + 'power_vs_time.png', bbox_inches='tight')

plt.show()