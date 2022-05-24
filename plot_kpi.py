import os
import matplotlib.pyplot as plt  
from system_params import * 

def create_img_folder(): 

    path = "{}/img/{}, V ={:.2e}, dth={:},lambda={:}/".format(
        os.getcwd(), opt_mode, LYA_V, D_TH, Amean)
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

