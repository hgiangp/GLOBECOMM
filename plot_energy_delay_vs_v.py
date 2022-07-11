import matplotlib.pyplot as plt
import pandas as pd # pandas is a dataframe library
import numpy as np 
from plot_kpi import access_img_folder
from system_params import * 


plt.rcParams.update({'font.family':'Helvetica'})

Amean = 9 
dth = 5 * Amean

csv_name = "result.csv"
Vs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

modes = ['Learning', 'Exhausted']

delay = np.zeros((len(modes), len(Vs)))
user_energy = np.zeros((len(modes), len(Vs)))
uav_energy = np.zeros((len(modes), len(Vs)))
weighted_energy2 = np.zeros((len(modes), len(Vs)))

for imode, mode in enumerate(modes): 
    opt_mode = mode
    for idx, V in enumerate(Vs):
        LYA_V = V
        path = access_img_folder(opt_mode=opt_mode, LYA_V=LYA_V, D_TH=dth)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[imode, idx] = np.array(data.delay)[:, np.newaxis][-1, 0]
        user_energy[imode, idx] = np.mean(data.energy_user)
        uav_energy[imode, idx] = np.mean(data.energy_uav)/no_users
        weighted_energy2[imode, idx] = PSI * uav_energy[imode, idx] + user_energy[imode, idx]

labels = ['Learning', 'Exhausted']
markers = ['s', 'o']
line_styles = ['-', '--'] 

weighted_energy_mW = weighted_energy2
delay_ms = delay*ts_duration*1e3

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))


# fig, (ax1, ax2) = plt.subplots(1, 2)

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}
ax1.semilogx(Vs, weighted_energy_mW[0, :], label=labels[0], marker=markers[0], linestyle=line_styles[0], color = 'tab:green')
ax1.semilogx(Vs, weighted_energy_mW[1, :], label=labels[1], marker=markers[1], linestyle=line_styles[1], color = 'tab:orange')
ax1.grid(True, which ="both", color = '0.7', ls = ":")
ax1.set_xlabel("Lyapunov Control Parameter V", fontsize=12)
ax1.set_ylabel("Average Power Consumption (mW)", fontsize=12)
ax1.set_yticks([200, 400, 600, 800])

ax2.semilogx(Vs, delay_ms[0, :], label=labels[0], marker=markers[0], linestyle=line_styles[0], color = 'tab:green')
ax2.semilogx(Vs, delay_ms[1, :], label=labels[1], marker=markers[1], linestyle=line_styles[1], color = 'tab:orange')
ax2.set_yticks([20, 30, 40, 50])
ax2.grid(True, which ="both", color = '0.7', ls = ":")
ax2.set_xlabel("Lyapunov Control Parameter V", fontsize=12)
ax2.set_ylabel("Average Latency (ms)",  fontsize=12)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol = 2, fontsize=12)
plt.savefig('./results/energy_delay_vs_V.png', bbox_inches='tight')
plt.savefig('./results/energy_delay_vs_V.eps', bbox_inches='tight')

plt.show()





# for imode in range(len(modes)): 
#     plt.plot(Vs, weighted_energy_mW[imode, :], label=labels[imode], marker=markers[imode], linestyle=line_styles[imode])

# plt.xscale(xscale)
# plt.xlabel(xlabelstr)
# plt.xticks(Vs)
# plt.grid(linestyle='--')
# plt.ylabel('Mean power consumption (W)')
# plt.xlim(Vs[0], Vs[-1])
# plt.legend()
# plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_power.png')
# plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_power.eps')
# plt.close()

# fig2 = plt.figure(2)
# delay_ms = delay*ts_duration*1000
# for imode in range(len(modes)):
#     plt.plot(Vs, delay_ms[imode, :], label=labels[imode], marker=markers[imode], linestyle=line_styles[imode])

# plt.xscale(xscale)
# plt.xlabel(xlabelstr)
# plt.xticks(Vs)
# plt.grid(linestyle='--')
# plt.xlim(Vs[0], Vs[-1])
# plt.ylabel('Average latency (ms)')
# plt.legend()
# plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_dth.png')
# plt.savefig(f'./{sub_path}/cmp_' + xlabelstr+'_vs_dth.eps')
# plt.close()


