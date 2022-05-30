from system_params import * 
import os 

def create_img_folder(): 

    path = "{}/img/{}, V ={:.2e}, psi = {:.3e}, dth={:},lambda={:}/".format(
        os.getcwd(), opt_mode, LYA_V, PSI, D_TH/Amean, Amean)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

path = create_img_folder()

import pickle

USERS_FILE = "users.pickle"
SERVER_FILE = "server.pickle"

def load_data_pickle(file_name): 
    with open(file_name, 'rb') as handle:
        object = pickle.load(handle)
    return object 



import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import DataFrame

def plot_moving_average(data, color, label, rolling_intv = 10): 
    df = pd.DataFrame(data)
    plt.plot(np.arange(len(data))+1, \
                np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), \
                color, label = label)

def plot_kpi_mva(data, path, title, color_list, label_list, ylabel=None, rolling_intv = 10): 
    for idata, dat in enumerate(data): 
        plot_moving_average(dat, color=color_list[idata], label=label_list[idata], rolling_intv=rolling_intv)
    
    plt.title(title)
    plt.legend()
    plt.xlabel('Time Frame')
    plt.ylabel(ylabel=ylabel)
    plt.grid()
    plt.savefig(path + title)
    # plt.show()
    plt.close()

def plot_users_kpi(users, path, rolling_intv = 10):
    kpi = 'PACKET'
    color = ['b', 'r', 'k', 'g']
    label_list = ['Arrival packets', 'Local packets', 'Offload packets', 'Queue length']
    for idx, user in enumerate(users):
        if kpi == 'PACKET':
            data_list = []
            data_list.append(user.A_i)
            data_list.append(user._a_i)
            data_list.append(user._b_i) 
            data_list.append(user.Q_i.value)
        elif kpi == 'ENERGY': 
            data_list = []
            pass 
        plot_kpi_mva(data_list, path, title='user[{}]'.format(idx), color_list=color, label_list=label_list)

def plot_server_kpi(servers, path, rolling_intv = 10):
    # color = ['b', 'r', 'k']
    # label_list = ['Offloaded packets', 'Computation packets', 'Queue length']
    color = ['b', 'r']
    label_list = ['Offloaded packets', 'Computation packets']
    
    for iuser in range(no_users): 
        data_list = []
        data_list.append(server._b_i[:, iuser])
        data_list.append(server._c_i[:, iuser])
        # queue = server.L_i[iuser].value.reshape(-1)
        # data_list.append(queue)
        plot_kpi_mva(data_list, path, title='uav_user[{}]'.format(iuser), color_list=color, label_list=label_list)

users = load_data_pickle(file_name=path + USERS_FILE)
server = load_data_pickle(file_name=path + SERVER_FILE)

plot_users_kpi(users=users, path=path, rolling_intv=50)
plot_server_kpi(servers=server, path = path, rolling_intv = 10)

print("finish")