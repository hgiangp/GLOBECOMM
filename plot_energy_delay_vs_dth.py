import matplotlib.pyplot as plt
import pandas as pd # pandas is a dataframe library
import numpy as np 
from plot_kpi import access_img_folder
from system_params import * 


plt.rcParams.update({'font.family':'Helvetica'})

LYA_V = 3*1e4
Amean = 9


csv_name = "result.csv"
dths = [3, 4, 5, 6, 7, 8]

mode = opt_mode_arr[0]

delay = np.zeros(len(dths))
user_energy = np.zeros(len(dths))
uav_energy = np.zeros(len(dths))
weighted_energy2 = np.zeros( len(dths))

for idx, dth in enumerate(dths):
    dth = dth * Amean
    path = access_img_folder(opt_mode=mode, D_TH=dth, LYA_V=LYA_V)
    
    file = path + csv_name
    data = pd.read_csv(file)
    delay[idx] = np.array(data.delay)[:, np.newaxis][-1, 0]
    user_energy[idx] = np.mean(data.energy_user)
    uav_energy[idx] = np.mean(data.energy_uav)/no_users
    weighted_energy2[idx] = PSI * uav_energy[idx] + user_energy[idx]

labels = ['Learning']
markers = ['s', 'o']
line_styles = ['-', '--'] 

weighted_energy_mW = weighted_energy2
delay_ms = delay*ts_duration*1e3
dths_ms = np.array(dths)*ts_duration*1e3

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

ax2.plot(dths_ms, delay_ms, marker=markers[0], linestyle=line_styles[0], color = 'tab:green')
# ax2.set_yticks([20, 30, 40, 50])
ax2.grid(True, which ="both", color = '0.8')
ax2.set_xlabel("Latency Threshold (ms)", fontsize=12)
ax2.set_ylabel("Average Latency (ms)",  fontsize=12)
ax2.set_xlim(30, 80)
# ax2.set_yticks([27, 30, 33, 36])



ax1.plot(dths_ms, weighted_energy_mW, marker=markers[0], linestyle=line_styles[0], color = 'tab:green')
ax1.grid(True, which ="both", color = '0.8')
ax1.set_xlabel("Latency Threshold (ms)", fontsize=12)
ax1.set_ylabel("Average Power Consumption (mW)", fontsize=12)
ax1.set_xlim(30, 80)
# ax1.set_yticks([240, 280, 320])



handles, labels = ax2.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol = 2, fontsize=12)
plt.savefig('./results/energy_delay_vs_dth.png', bbox_inches='tight')
plt.savefig('./results/energy_delay_vs_dth.eps', bbox_inches='tight')

plt.show()


# fig, ax1 = plt.subplots()
# ax1.plot(dths_ms, weighted_energy_mW, marker=markers[0], linestyle=line_styles[0], color = 'tab:green')

# ax1.set_xticks([30, 40, 50, 60, 70, 80])
# ax1.set_yticks([220, 260, 300, 340])
# ax1.set_ylabel('Average Power Comsumption (mW)')
# ax1.grid(linestyle='--')
# ax1.tick_params(axis='y')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel('Average system delay (ms)')  # we already handled the x-label with ax1
# # ax2.plot(t, delay[0, :], color=color_learning)
# # ax2.plot(t, delay[1, :], color=color_exhauted)
# ax2.plot(dths_ms, delay_ms, '-s')

# # ax2.grid(linestyle='--')
# ax2.tick_params(axis='y')
# # ax1.set_xlim(dt[0], dth_arr[-1])
# # ax2.set_xscale('log')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('./results/energy_delay_vs_dth.eps')
# plt.show()
# plt.close()