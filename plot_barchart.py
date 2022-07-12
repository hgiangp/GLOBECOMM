import numpy as np
import matplotlib.pyplot as plt
import itertools

import pandas
from plot_kpi import access_img_folder
from system_params import *

plt.rcParams.update({'font.family':'Helvetica'})

# plt.rcParams.update({'font.size': 12})
  

width = 0.25
space = 0.03
fig, ax = plt.subplots()
ax.grid(True, axis = 'y', color = '0.8', linestyle = '-')


####### Read data from pickle file #######

users_num = [8, 10, 12, 15]

N = len(users_num)
ind = np.arange(N) 

greedy_vals = []
learning_vals = []
exhausted_vals = []
csv_file_name = "result.csv"

for index, no_users in enumerate(users_num): 
    ############## Greedy ###############

    path_file = access_img_folder(opt_mode='greedy', no_users=no_users) + csv_file_name
    data = pandas.read_csv(path_file)
    ue_power = np.mean(data.energy_user)
    uav_power = np.mean(data.energy_uav)/no_users
    greedy_vals.append(ue_power + PSI * uav_power)

    ############## Learning ###############

    path_file = access_img_folder(opt_mode='learning', no_users=no_users) + csv_file_name
    data = pandas.read_csv(path_file)
    ue_power = np.mean(data.energy_user)
    uav_power = np.mean(data.energy_uav)/no_users
    learning_vals.append(ue_power + PSI * uav_power)

    ############## Exhausted ###############

    path_file = access_img_folder(opt_mode='exhausted', no_users=no_users) + csv_file_name
    data = pandas.read_csv(path_file)
    ue_power = np.mean(data.energy_user)
    uav_power = np.mean(data.energy_uav)/no_users
    exhausted_vals.append(ue_power + PSI * uav_power)
  
# greedy_vals = [720, 771, 812, 866]
bar1 = ax.bar(ind, greedy_vals, width, color = 'none', hatch= 'xx', edgecolor = 'tab:blue', linewidth = 1.5 )
  
# learning_vals = [460, 535, 591, 659]
bar2 = ax.bar(space + ind+width, learning_vals, width, color = 'none', hatch= '\\\\', edgecolor='tab:green', linewidth = 1.5)
  
# exhausted_vals = [432, 456, 515, 563]
bar3 = ax.bar(space*2 + ind+width*2, exhausted_vals, width,  color = 'none', hatch = '//', edgecolor = 'tab:orange', linewidth = 1.5)
  
ax.set_xlabel("Number of IDs", fontsize = 12)
ax.set_ylabel('Average Power Consumption (mW)', fontsize = 12)
ax.set_ylim(100, 950)
ax.set_yticks([200, 400, 600, 800])
# plt.grid(True, axis = 'y', color = '0.6', linestyle = '-')

# plt.title("Players Score")
  

xsticks = [str(no_users) for no_users in users_num]

ax.set_xticks(ind+width+space, xsticks)
ax.legend( (bar1, bar2, bar3), ('Max-Queue', 'Learning', 'Exhausted'), handlelength = 2, handleheight = 2, fontsize = 12)

rects = ax.patches

# Make some labels
labels = [str(int(x)) for x in itertools.chain(greedy_vals, learning_vals, exhausted_vals)]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
    )


plt.savefig('./results/energy_vs_no_users.png', bbox_inches='tight')
plt.savefig('./results/energy_vs_no_users.eps', bbox_inches='tight')
plt.show()