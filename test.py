from system_params import * 
import os 

import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import DataFrame

def create_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, PSI=PSI, D_TH=D_TH, Amean=Amean, delta_t=delta_t, no_users=no_users): 
    sub_path = 'img'
    path = "{}/{}/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}, no_user={}, delta_t={}/".format(
        os.getcwd(), sub_path, opt_mode, LYA_V, PSI, D_TH/Amean, Amean, no_users, delta_t)
    print(path)
    os.makedirs(path, exist_ok=True)
    return path


import pickle

def load_data_pickle(file_name): 
    with open(file_name, 'rb') as handle:
        object = pickle.load(handle)
    return object 

def plot_moving_average(data, color, label, rolling_intv = 10): 
    
    data_arr = np.asarray(data)
    df = pd.DataFrame(data)

    # plt.plot(np.arange(len(data_arr))+1, data_arr)
    plt.plot(np.arange(len(data))+1, \
                np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), \
                color, label = label)
    # plt.fill_between(np.arange(len(data_arr))+1,\
    #     np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), \
    #     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), \
    #         color = 'b', alpha = 0.2)
    pass 

def plot_kpi_mva(data, path, title, color_list, label_list, ylimit = None, ylabel=None, rolling_intv = 10): 
    for idata, dat in enumerate(data): 
        plot_moving_average(dat, color=color_list[idata], label=label_list[idata], rolling_intv=rolling_intv)

    plt.title(title)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(ylabel=ylabel)
    plt.ylim(ylimit)
    plt.grid()
    plt.savefig(path + title)
    # plt.show()
    plt.close()

####################
# save data 
####################
arrival_packets_arr = np.zeros((no_slots, no_users))
local_packets_arr = np.zeros((no_slots, no_users))
off_packets_arr = np.zeros((no_slots, no_users))

ue_pro_power_arr = np.zeros((no_slots, no_users))
ue_off_power_arr = np.zeros((no_slots, no_users))
ue_total_power_arr = np.zeros((no_slots, no_users))
weighted_power_arr_mW = np.zeros((no_slots))
ue_queue_arr = np.zeros((no_slots, no_users))
ue_virtual_arr = np.zeros((no_slots, no_users))
ue_delay_arr = np.zeros((no_slots, no_users))
ue_drift_arr = np.zeros((no_slots, no_users))

uav_packets_arr = np.zeros((no_slots, no_users))
uav_queue_arr = np.zeros((no_slots, no_users))

uav_power_arr = np.zeros((no_slots))
weighted_power_arr_mW = np.zeros((no_slots)) 
ue_power_arr = np.zeros((no_slots, 1))


def load_data(path, users, server):
    global uav_power_arr, ue_power_arr, weighted_power_arr_mW
    ######### User ########
    for iuser, user in enumerate(users): 
        arrival_packets_arr[:, iuser] = user.A_i[:, 0]
        local_packets_arr[:, iuser] = user._a_i[:, 0]
        off_packets_arr[:, iuser] = user._b_i[:, 0]
        ue_pro_power_arr[:, iuser] = user._pro_energy[:, 0]/ts_duration*1000 # mW 
        ue_off_power_arr[:, iuser] = user._off_energy[:, 0]/ts_duration*1000 # mW
        ue_total_power_arr[:, iuser] = ue_pro_power_arr[:, iuser] + ue_off_power_arr[:, iuser]
        ue_queue_arr[:, iuser] = user.Q_i.value[:, 0]
        ue_delay_arr[:, iuser] = user._delay[:, 0]
        ue_virtual_arr[:, iuser] = user._virtual_queue[:, 0]
        ue_drift_arr[:, iuser] = user.drift[:, 0] # drift in TS 
    #######UAV #########
    for iuser in range(no_users): 
        uav_packets_arr[:, iuser] = server._c_i[:, iuser]
        uav_queue_arr[:, iuser] = server.L_i[iuser].value[:, 0]

    uav_power_arr = server._energy[:, 0]/ts_duration*1000 # mW 

    ue_power_arr = (np.sum(ue_total_power_arr, axis=1))
    weighted_power_arr_mW = ((ue_power_arr + PSI * uav_power_arr)) # mW 


def plot_gain(users): 
    step = 10
    gains = todB(users[0].gain)
    gains_slice = gains[::step]
    x = np.arange(gains.shape[0])
    x_slide = x[::step]
    print(gains.shape)

    plt.plot(x_slide, gains_slice)
    print(np.mean(gains))
    plt.show()

def plot_drift(path, rolling_intv = 10): 
    'plot drift and V*E of each user'
    fig = plt.figure()
    scale = 1 # scale to 1e6 since V * E is too large to plot
    data_list = []
    data_list.append(np.mean(ue_drift_arr, axis=1))
    data_list.append(LYA_V * mW(weighted_power_arr_mW) * ts_duration * scale/no_users)

    label_list = ['average drift', 'average scale weighted energy, scale = {}'.format(scale)]
    plot_kpi_mva(data_list, path, title='average_drift', color_list=['b', 'r'], label_list=label_list, rolling_intv=rolling_intv)


def plot_users_kpi(users, path, kpi = 'packet', rolling_intv = 10):
    '''
    Plot each user 
    '''

    for idx, user in enumerate(users):
        if kpi == 'packet':
            color = ['b', 'r', 'k', 'g']
            label_list = ['Arrival packets', 'Local packets', 'Offload packets', 'Queue length']
            data_list = []
            data_list.append(user.A_i)
            data_list.append(user._a_i)
            data_list.append(user._b_i) 
            data_list.append(user.Q_i.value)
        elif kpi == 'energy': 
            data_list = []
            color = ['b', 'r', 'g']
            label_list = ['Local processing power', 'Offloading power', 'Total power (mW)']
            data_list.append(ue_pro_power_arr[:, idx])
            data_list.append(ue_off_power_arr[:, idx]) 
            data_list.append(ue_total_power_arr[:, idx]) 
        plot_kpi_mva(data_list, path, title='user_{}[{}]'.format(kpi, idx), color_list=color, label_list=label_list, rolling_intv=rolling_intv)

def plot_server_kpi(servers, path, rolling_intv = 10):
    is_plot_queue_length = True 
    if is_plot_queue_length: 
        color = ['b', 'r', 'g']
        label_list = ['Offloaded packets', 'Computation packets', 'Queue length']
    else: 
        color = ['b', 'r']
        label_list = ['Offloaded packets', 'Computed packets']

    for iuser in range(no_users): 
        data_list = []
        data_list.append(server._b_i[:, iuser])
        data_list.append(server._c_i[:, iuser])
        if is_plot_queue_length: 
            data_list.append(uav_queue_arr[:, iuser])
        
        plot_kpi_mva(data_list, path, title='uav_user[{}]'.format(iuser), color_list=color, label_list=label_list, rolling_intv=rolling_intv)

def plot_ue_power(path, rolling_intv = 10): 
    'plot average processing, offloading, total energy at UE'
    fig = plt.figure()
    data_list = []
    data_list.append(np.mean(ue_pro_power_arr, axis=1))
    data_list.append(np.mean(ue_off_power_arr, axis=1))
    # data_list.append(np.mean(ue_total_power_arr, axis=1))
    # label_list = ['UE processing power (mW)', 'UE offloading power (mW)', 'UE total power (mW)']
    label_list = ['UE processing power (mW)', 'UE offloading power (mW)']
    plot_kpi_mva(data_list, path, title='average_user_power', color_list=['b', 'r', 'k'], label_list=label_list, rolling_intv=rolling_intv)

def plot_weighted_power(path, rolling_intv = 10): 
    'plot ue total power, uav total power, weighted power'
    fig = plt.figure()
    data_list = []
    data_list.append(ue_power_arr/no_users)
    data_list.append(uav_power_arr/no_users)
    data_list.append(weighted_power_arr_mW/no_users)
    label_list = ['UE power (mW)', 'UAV power (mW)', 'Weighted power (mW)']
    plot_kpi_mva(data_list, path, title='average_weighted_power', color_list=['b', 'r', 'g'], label_list=label_list, rolling_intv=rolling_intv)

def plot_server_tasks(path, rolling_intv = 10): 
    # 'plot server offloaded tasks, computed tasked', 'queue length'
    'plot server offloaded tasks, computed tasked'
    fig = plt.figure()
    data_list = []
    data_list.append(np.mean(off_packets_arr, axis=1))
    data_list.append(np.mean(uav_packets_arr, axis=1))
    # data_list.append(np.mean(uav_queue_arr, axis=1))
    # label_list = ['UAV arrival packets (offloaed)', 'UAV computed packets', 'UAV queue length']
    label_list = ['UAV arrival packets (offloaed)', 'UAV computed packets']
    plot_kpi_mva(data_list, path, title='average_uav_packets', color_list=['b', 'r', 'g'], label_list=label_list, rolling_intv=rolling_intv)

def plot_users_tasks(path, rolling_intv = 10): 
    'plot user arrival, computed tasked, offloaded, queue length'
    fig = plt.figure()
    data_list = []
    data_list.append(np.mean(arrival_packets_arr, axis=1))
    data_list.append(np.mean(local_packets_arr, axis=1))
    data_list.append(np.mean(off_packets_arr, axis=1))
    data_list.append(np.mean(ue_queue_arr, axis=1))

    label_list = ['UE arrival packets', 'UE local computed packets','UE offloaded packets', 'UE queue length']
    plot_kpi_mva(data_list, path, title='average_ue_packets', color_list=['b', 'r', 'k', 'g'], label_list=label_list, rolling_intv=rolling_intv)

def plot_virtual_queue(path, rolling_intv = 10): 
    fig = plt.figure()
    data_list = []
    data_list.append(np.mean(ue_virtual_arr, axis=1))
    label_list = ['Virtual queue']
    color_list=['b']
    plot_kpi_mva(data_list, path, title='average_ue_virtual_queue', color_list=color_list, label_list=label_list, rolling_intv=rolling_intv)

def plot_delay(path, rolling_intv = 10): 
    fig = plt.figure()
    data_list = []
    data_list.append(np.mean(ue_delay_arr, axis=1))
    label_list = ['delay']
    color_list=['b']
    # ylimit = [0, 10]
    plot_kpi_mva(data_list, path, title='average_ue_delay', color_list=color_list, label_list=label_list, rolling_intv=rolling_intv)

    # plot_kpi_mva(data_list, path, title='average_ue_delay', color_list=color_list, label_list=label_list,  ylimit=ylimit, rolling_intv=rolling_intv)

def plot_users_delay(path, rolling_intv = 10):
    '''
    Plot each user 
    '''
    label_list = []
    import random


    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(no_users)]
    
    data_list = []
    for iuser in range(no_users):
            data_list.append(ue_delay_arr[:, iuser])
            label_list.append('user[{}]'.format(iuser))

    plot_kpi_mva(data_list, path, title='users_delay'.format(iuser), color_list=color, label_list=label_list, rolling_intv=rolling_intv)


def plot_optimizer_offloading_decision(optimizer, path, rolling_intv = 10):
    import matplotlib.pyplot as plt  
        # plot number of offloading users 
    average_offloading_users = np.sum(optimizer.mode_his, axis = 1)

    print('average offloaing users: ' + str(np.mean(optimizer.mode_his, axis = 0)))
    print("average delay: " + str(optimizer.delay[-1]))
    
    data = average_offloading_users
    rolling_intv = 10 
    df = pd.DataFrame(data)
    plt.plot(np.arange(len(data))+1, \
                np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), \
                color = 'b', label = 'average nums of offloading UEs')
    plt.xlabel('Time')
    plt.grid(True)
    plt.savefig(path + 'offloading_UEs')


if __name__ == '__main__':
    path = create_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, PSI=PSI, D_TH=D_TH, Amean=Amean)
    
    users = load_data_pickle(file_name=path + USERS_FILE)
    server = load_data_pickle(file_name=path + SERVER_FILE)

    # optimizer = load_data_pickle(file_name=path + OPTIMIZER_FILE)
    # plot_optimizer_offloading_decision(optimizer, path)

    load_data(path, users, server)
    rolling_intv = 50
    plot_users_kpi(users=users, path=path, kpi='packet', rolling_intv=rolling_intv)
    plot_users_kpi(users=users, path=path, kpi='energy', rolling_intv=rolling_intv)
    plot_server_kpi(servers=server, path = path, rolling_intv = rolling_intv)
    plot_ue_power(path=path, rolling_intv= rolling_intv)
    plot_weighted_power(path=path, rolling_intv= rolling_intv)
    plot_server_tasks(path, rolling_intv = rolling_intv)
    plot_users_tasks(path, rolling_intv = rolling_intv)
    plot_virtual_queue(path, rolling_intv = rolling_intv)
    plot_delay(path, rolling_intv = rolling_intv)
    plot_users_delay(path, rolling_intv = rolling_intv)
    plot_drift(path, rolling_intv = rolling_intv)
    print("finish")

# plot_gain(users=users)


