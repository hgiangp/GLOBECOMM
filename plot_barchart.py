import numpy as np
import matplotlib.pyplot as plt
import itertools

plt.rcParams.update({'font.family':'Helvetica'})

# plt.rcParams.update({'font.size': 12})
  
N = 4
ind = np.arange(N) 
width = 0.25
space = 0.03
fig, ax = plt.subplots()
ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')

  
greedy_vals = [720, 771, 812, 866]
bar1 = ax.bar(ind, greedy_vals, width, color = 'none', hatch= 'xx', edgecolor = 'tab:blue', linewidth = 1.5 )
  
learning_vals = [460, 535, 591, 659]
bar2 = ax.bar(space + ind+width, learning_vals, width, color = 'none', hatch= '\\\\', edgecolor='tab:green', linewidth = 1.5)
  
exhausted_vals = [432, 456, 515, 563]
bar3 = ax.bar(space*2 + ind+width*2, exhausted_vals, width,  color = 'none', hatch = '//', edgecolor = 'tab:orange', linewidth = 1.5)
  
ax.set_xlabel("Number of IDs", fontsize = 12)
ax.set_ylabel('Average Power Consumption (mW)', fontsize = 12)
ax.set_ylim(400, 1000)
# plt.grid(True, axis = 'y', color = '0.6', linestyle = '-')

# plt.title("Players Score")
  
ax.set_xticks(ind+width+space,['8', '10', '12', '15'])
ax.legend( (bar1, bar2, bar3), ('Max-Queue', 'Learning', 'Exhausted'), handlelength = 2, handleheight = 2, fontsize = 12)

rects = ax.patches

# Make some labels
labels = [x for x in itertools.chain(greedy_vals, learning_vals, exhausted_vals)]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
    )


plt.savefig('./results/energy_vs_no_users.png', bbox_inches='tight')
plt.savefig('./results/energy_vs_no_users.eps', bbox_inches='tight')
plt.show()