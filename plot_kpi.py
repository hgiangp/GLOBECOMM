import pandas as pd 
import os
import matplotlib.pyplot as plt  
from system_params import * 

def create_img_folder(): 

    path = "{}/img/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}/".format(
        os.getcwd(), opt_mode, LYA_V, PSI, D_TH, Amean)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

def plot_kpi_users(data_list, kpi_list, path, title): 
    n_ts = data_list[0].shape[0]
    fig = plt.figure()
    for (kpi, data) in zip(kpi_list, data_list): 
        plt.plot(np.arange(n_ts), data, label=kpi)
    
    plt.title(title)
    plt.xlabel('Time frames')
    plt.legend()
    plt.grid()
    plt.savefig(path + title)
    # plt.show()
    plt.close()

def plot_rate( rate_his, rolling_intv = 50, ylabel='Normalized Computation Rate', name='Average queue length'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    fig = plt.figure()
    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    plt.grid()
    plt.plot(np.arange(len(rate_array))+1, rate_array)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.savefig(name)
    plt.show()
    plt.close()

def plot_kpi_avr(data_list, kpi_list, path): 
    rolling_intv = 10
    for (kpi, data) in zip(kpi_list, data_list): 
        plot_rate(data, rolling_intv, kpi, path+kpi)

def plot_kpi_data(xdata, xlabel, xscale, path):
    no_elements = len(xdata)

    queue = np.size(no_elements, 2)
    energy = np.size(no_elements, 3)
    tasks = np.size(no_elements, 3)
    delay = np.size(no_elements, 1)
    for idata, data in xdata:
        V = data
        path = "{}/img/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}/".format(
        os.getcwd(), opt_mode, LYA_V, PSI, D_TH, Amean)
        
        queue[idata, :], energy[idata, :], tasks[idata, :], delay[idata, :] = load_data(path=path)
    # plot_queue


def plot_delay(): 

    csv_name = "result.csv"
    # dth_arr = [10, 15, 20, 40]
    # xscale = 'linear'
    # xlabelstr = 'Delay threshold'

    dth_arr = [1e3, 1e4, 1e5, 2*1e5, 3*1e5, 4*1e5, 5*1e5]
    xscale = 'log'
    xlabelstr = 'Lyapunov control parameter, V' 
    
    delay = np.zeros(len(dth_arr))
    user_energy = np.zeros(len(dth_arr))
    uav_energy = np.zeros(len(dth_arr))
    weighted_energy2 = np.zeros(len(dth_arr))
    user_queue, uav_queue = np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    localA, offloadB, remoteC = np.zeros(len(dth_arr)), np.zeros(len(dth_arr)), np.zeros(len(dth_arr))

    


    for idx, dth in enumerate(dth_arr):
        # d_th = dth_arr[idx]
        # scale_delay = dth_arr[idx]
        LYA_V = dth

        path = "{}/img/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}/".format(
        os.getcwd(), opt_mode, LYA_V, PSI, D_TH, Amean)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[idx] = np.mean(data.delay)
        user_energy[idx] = np.mean(data.energy_user)*1000/ts_duration
        uav_energy[idx] = np.mean(data.energy_uav)*1000/ts_duration
        weighted_energy2[idx] = np.mean(data.weightedE)* 1000/ts_duration
        user_queue[idx] = np.mean(data.local_queue)
        uav_queue[idx] = np.mean(data.uav_queue)
        localA[idx] = np.mean(data.local_a)
        offloadB[idx] = np.mean(data.off_b)
        remoteC[idx] = np.mean(data.remote_c)

        # weighted_energy[idx] = np.mean(data.aweightedE)*1000/ts_duration
    plt.plot(dth_arr, user_energy, '-ob', label='User power')
    plt.plot(dth_arr, uav_energy, '-or', label='Weighted UAV power')        
    plt.plot(dth_arr, weighted_energy2, '-ok', label='Weighted power')

    plt.xlabel(xlabelstr)
    plt.xticks(dth_arr)
    plt.xscale(xscale)
    plt.ylabel('Power consumption (mW)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_power.png')
    plt.show()


    # plt.plot(dth_arr, dth_arr - delay, '-o', label = "ts_duration delay (TS)")
    plt.plot(dth_arr, delay, '-ob', label = "Delay")
    
    plt.xscale(xscale)
    plt.xticks(dth_arr)

    plt.xlabel(xlabelstr)
    plt.ylabel('Delay (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_dth.png')
    plt.show()

    scale = 1
    plt.plot(dth_arr, localA/scale, '-ob', label = "Local computation packets")
    plt.plot(dth_arr, offloadB/scale, '-or', label = "Offloading packets")
    plt.plot(dth_arr, remoteC/scale, '-.ok', label = "UAV computation packets")

    plt.xscale(xscale)
    plt.xticks(dth_arr)
    plt.xlabel(xlabel=xlabelstr)
    plt.ylabel('Computation and offload volume (packets)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_abc.png')
    plt.show()

    plt.plot(dth_arr, user_queue, '-ob', label = "User queue")
    plt.plot(dth_arr, uav_queue, '-or', label = "UAV queue")
    plt.xlabel(xlabel=xlabelstr)
    plt.xticks(dth_arr)

    plt.xscale(xscale)
    plt.ylabel('Queue length (packets)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_queue.png')
    # plt.show()

def load_data(path, file_name='result.csv'):
    data = pd.read_csv(path+file_name)
    # df = pd.DataFrame(
    # {'local_queue':Q_mean,'uav_queue':L_mean,
    # 'energy_user':E_ue_mean,'energy_uav':E_uav_mean, 
    # 'delay':ava_delay, 'weightedE': W_E_mean, 
    # 'off_b': b_mean, 'local_a': a_mean, 'remote_c': c_mean, 
    # # 'time': total_time
    # })
    Q_mean = np.mean(data.local_queue)
    L_mean = np.mean(data.uav_queue)
    E_ue = np.mean(data.energy_user)
    E_uav = np.mean(data.energy_uav)
    delay = np.mean(data.ava_delay)
    WeightedE = np.mean(data.W_E_mean)
    local_a = np.mean(data.local_a)
    off_b = np.mean(data.off_b)
    remote_c = np.mean(data.remote_c)

    queue_info = (Q_mean, L_mean)
    E_info = (E_ue, E_uav, WeightedE)
    tasks = (local_a, off_b, remote_c)

    return queue_info, E_info, tasks, delay 


# plot_delay()





