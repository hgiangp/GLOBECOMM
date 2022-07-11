import pandas as pd 
import os
import matplotlib.pyplot as plt  
from system_params import *


sub_path = 'img'
no_users = 10 # number of users

def access_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, PSI=PSI, D_TH=D_TH, Amean=Amean, no_users=no_users, delta_t=delta_t): 

    path = "{}/{}/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}, no_user={}, delta_t={}/".format(
        os.getcwd(), sub_path, opt_mode, LYA_V, PSI, D_TH/Amean, Amean, no_users, delta_t)
    return path 

def plot_twin(power, delay, dth_arr, xlabel = 'Lyapunov parameter V', xscale = 'log', num_element = 2): 
    # Create some mock data
    t = dth_arr # Values of LYA_V
    delay = delay * ts_duration * 1000 # convert to ms 
    fig, ax1 = plt.subplots()

    color_learning = 'tab:red'
    color_exhauted = 'tab:blue'

    ax1.set_xscale(xscale)

    ax1.plot(t, power[0, :], '-s')
    if num_element == 2: 
        ax1.plot(t, power[1, :], '-s')

    ax1.set_xlim(dth_arr[0], dth_arr[-1])
    ax1.set_xticks(t)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Average power comsumption (mW)')

    # ax1.plot(t, power[0, :], color=color_learning)
    # ax1.plot(t, power[1, :], color=color_exhauted)

    ax1.grid(linestyle='--')

    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average system delay (ms)')  # we already handled the x-label with ax1
    ax2.plot(t, delay[0, :], '-s')
    if num_element == 2:
        ax2.plot(t, delay[1, :], '-s')

    # ax2.grid(linestyle='--')
    ax2.tick_params(axis='y')
    ax1.set_xlim(dth_arr[0], dth_arr[-1])
    # ax2.set_xscale('log')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('img/power_delay.eps')
    plt.show()
    plt.close()

def plot_delay(): 
    csv_name = "result.csv"
    dth_arr = [5*1e2, 2*1e3, 1e4, 5*1e4, 2*1e5, 5*1e5, 1e6]

    # xscale = 'linear'
    # xlabelstr = 'Delay threshold'

    # dth_arr =  [1e3, 5*1e3, 1e4,5*1e4, 1e5, 5*1e5, 1e6]
    xscale = 'log'
    xlabelstr = 'Lyapunov control parameter, V' 

    modes = ['Learning', 'Exhausted']
    
    delay = np.zeros((len(modes), len(dth_arr)))
    user_energy = np.zeros((len(modes), len(dth_arr)))
    uav_energy = np.zeros((len(modes), len(dth_arr)))
    weighted_energy2 = np.zeros((len(modes), len(dth_arr)))
    user_queue, uav_queue = np.zeros((len(modes), len(dth_arr))), np.zeros((len(modes), len(dth_arr)))
    localA, offloadB, remoteC = np.zeros((len(modes), len(dth_arr))),np.zeros((len(modes), len(dth_arr))), np.zeros((len(modes), len(dth_arr)))
    for imode, mode in enumerate(modes): 
        opt_mode = mode
        for idx, d_th in enumerate(dth_arr):
            LYA_V = d_th
            # D_TH = d_th*Amean
            path = access_img_folder(opt_mode=opt_mode, LYA_V=LYA_V)
            
            file = path + csv_name
            data = pd.read_csv(file)
            # delay[imode, idx] = np.mean(data.delay)
            delay[imode, idx] = np.array(data.delay)[:, np.newaxis][-1, 0]
            user_energy[imode, idx] = np.mean(data.energy_user)
            # uav_energy[idx] = np.mean(data.energy_uav)
            # weighted_energy2[idx] = np.mean(data.weightedE)
            uav_energy[imode, idx] = np.mean(data.energy_uav)/no_users
            weighted_energy2[imode, idx] = PSI * uav_energy[imode, idx] + user_energy[imode, idx]
            user_queue[imode, idx] = np.mean(data.local_queue)
            uav_queue[imode, idx] = np.mean(data.uav_queue)
            localA[imode, idx] = np.mean(data.local_a)
            offloadB[imode, idx] = np.mean(data.off_b)
            remoteC[imode, idx] = np.mean(data.remote_c)
    
    plot_twin(weighted_energy2, delay, dth_arr)
    fig1 = plt.figure(1)
    
    # weighted_energy[idx] = np.mean(data.aweightedE)*1000/ts_duration
    # labels = ['learning', 'exhausted search', 'random']
    # markers = ['s', 'o', 'v']
    # line_styles = ['-', '--', '-.']
    labels = ['Learning', 'Exhausted']
    markers = ['s', 'o']
    line_styles = ['-', '--']    
    weighted_energy_W = weighted_energy2/1000
    for imode in range(len(modes)): 
        plt.plot(dth_arr, weighted_energy_W[imode, :], label=labels[imode], marker=markers[imode], linestyle=line_styles[imode])

    plt.xscale(xscale)
    plt.xlabel(xlabelstr)
    plt.xticks(dth_arr)
    plt.grid(linestyle='--')
    plt.ylabel('Mean power consumption (W)')
    plt.xlim(dth_arr[0], dth_arr[-1])
    plt.legend()
    plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_power.png')
    plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_power.eps')
    plt.close()

    fig2 = plt.figure(2)
    delay_ms = delay*ts_duration*1000
    for imode in range(len(modes)):
        plt.plot(dth_arr, delay_ms[imode, :], label=labels[imode], marker=markers[imode], linestyle=line_styles[imode])

    plt.xscale(xscale)
    plt.xlabel(xlabelstr)
    plt.xticks(dth_arr)
    plt.grid(linestyle='--')
    plt.xlim(dth_arr[0], dth_arr[-1])
    plt.ylabel('Average latency (ms)')
    plt.legend()
    plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_dth.png')
    plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_dth.eps')
    plt.close()


def plot_delay_bk(): 
    csv_name = "result.csv"
    # dth_arr = [3, 4, 5, 6, 7, 8]
    # xscale = 'linear'
    # xlabelstr = 'Latency threshold'

    dth_arr = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    xscale = 'log'
    xlabelstr = 'Lyapunov control parameter, V' 

    # modes = [opt_mode[0], opt_mode[1]]
    
    delay = np.zeros(len(dth_arr))
    user_energy = np.zeros(len(dth_arr))
    uav_energy = np.zeros( len(dth_arr))
    weighted_energy2 = np.zeros(len(dth_arr))
    user_queue, uav_queue = np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    localA, offloadB, remoteC = np.zeros(len(dth_arr)), np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    for idx, d_th in enumerate(dth_arr):
        # D_TH=d_th*Amean
        LYA_V = d_th
        path = access_img_folder(opt_mode = 'learning', LYA_V=LYA_V, D_TH=D_TH)
        
        file = path + csv_name
        data = pd.read_csv(file)
        # delay[idx] = np.mean(data.delay)
        delay[idx] = np.array(data.delay)[:, np.newaxis][-1, 0]
        user_energy[idx] = np.mean(data.energy_user)
        uav_energy[idx] = np.mean(data.energy_uav)/no_users
        weighted_energy2[idx] = PSI * uav_energy[idx] + user_energy[idx]
        user_queue[idx] = np.mean(data.local_queue)
        uav_queue[idx] = np.mean(data.uav_queue)
        localA[idx] = np.mean(data.local_a)
        offloadB[idx] = np.mean(data.off_b)
        remoteC[idx] = np.mean(data.remote_c) 

    weighted_energy2 = weighted_energy2.reshape(1, -1)
    delay = delay.reshape(1, -1)


    plot_twin(weighted_energy2, delay, dth_arr, xlabel=xlabelstr, xscale=xscale, num_element = 1)

    weighted_energy2 = weighted_energy2.flatten()
    delay = delay.flatten()

    fig1 = plt.figure(1)
    # plt.plot(dth_arr, user_energy, '-ob', label='User power')
    # plt.plot(dth_arr, uav_energy, '-or', label='UAV power')        
    plt.plot(dth_arr, weighted_energy2, '-ob', label='Weighted power')
    plt.xscale(xscale)
    # plt.xlim(1e3, 5*1e5)
    plt.xlabel(xlabelstr)
    plt.xticks(ticks= dth_arr)
    plt.grid(linestyle='--')
    plt.ylabel('Average system power consumption (mW)')
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_power.png')
    plt.savefig('./img/' + xlabelstr+'_vs_power.eps')
    plt.close()


    fig2 = plt.figure(2)
    # dth_ms = np.array(dth_arr)*10 # convert to ms
    dth_ms = dth_arr
    delay_ms = delay*ts_duration*1000
    gaps = dth_ms - delay_ms
    print(gaps)
    plt.plot(dth_ms, delay_ms, '-ob', label="Delay")

    plt.xscale(xscale)
    plt.xlabel(xlabelstr)
    # plt.xticks(dth_arr)
    plt.grid(linestyle='--')
    # plt.xlim(dth_arr[0], dth_arr[-1])
    plt.ylabel('Average latency (ms)')
    plt.legend()
    plt.savefig('./img/'+ xlabelstr +'_vs_dth.png')
    plt.savefig('./img/'+ xlabelstr +'_vs_dth.eps')
    plt.close()

    fig3 = plt.figure(3)
   
    scale = 1
    plt.plot(dth_arr, localA/scale, '-ob', label = "Local computation packets")
    plt.plot(dth_arr, offloadB/scale, '-or', label = "Offloading packets")
    plt.plot(dth_arr, remoteC/scale, '-.ok', label = "UAV computation packets")
    plt.xscale(xscale)
    plt.xlabel(xlabel=xlabelstr)
    plt.grid(linestyle='--')
    plt.xticks(dth_arr)
    plt.ylabel('Computation and offload volume (packets)')
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_abc.png')
    plt.savefig('./img/' + xlabelstr + '_vs_abc.eps')
    plt.close()
    # plt.show()

    fig4 = plt.figure(4)

    plt.plot(dth_arr, user_queue, '-ob', label = "User queue")
    plt.plot(dth_arr, uav_queue, '-or', label = "UAV queue")
    plt.xlabel(xlabel=xlabelstr)
    plt.xscale(xscale)
    plt.grid(linestyle='--')
    plt.xticks(dth_arr)
    plt.ylabel('Queue length (packets)')
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_queue.png')
    plt.savefig('./img/' + xlabelstr + '_vs_queue.eps')
    plt.close()

def plot_barchart(): 
    labels = ['15']
    # labels = ['8', '10', '12']
    # users_num_arr = [8, 10, 12]
    power_means = []
    # search_algs = ['random', 'learning', 'exhausted']
    search_algs = ['greedy', 'learning', 'exhausted']
    hatchs = ['x', '\\', '/']

    ###### GET PARAMS
    csv_name = "result.csv"
    for ialg, alg in enumerate(search_algs):
        no_users = 15
        opt_mode = alg 
        path = access_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, no_users=no_users)
        data = pd.read_csv(path + csv_name)
        power_means.append((np.mean(data.energy_user) + PSI * np.mean(np.mean(data.energy_uav)/no_users)))
    ###### PLOTS 


    x = np.arange(len(labels))  # the label locations
    width = 0.00005  # the width of the bars

    # plt.figure(figsize=(1,1),dpi=300)
    fig, ax = plt.subplots(figsize=(4, 4))
    for ialg, alg in enumerate(search_algs): 
        rects1 = ax.bar(x + (ialg - 1)*width, power_means[ialg], width, label=alg, hatch=hatchs[ialg], fill=False)
        ax.bar_label(rects1, padding=3)


    # rects1 = ax.bar(x - width, power_means[0], width, label=search_algs[0], hatch=hatchs[0], fill=False)
    # rects2 = ax.bar(x, power_means[1], width, label=search_algs[1], hatch=hatchs[1], fill=False)
    # rects3 = ax.bar(x + width, power_means[2], width, label=search_algs[2], hatch=hatchs[2], fill=False)

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Power (mW)')
    ax.set_xlabel('Number of UEs')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'./img/power_vs_no_users{no_users}.png', dpi=300)
    plt.show()

# plot_barchart()
# plot_delay_bk()
# plot_delay()
