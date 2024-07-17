from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns


x_1 = [0.05,0.1,0.2,0.3,0.4,0.5] 
x1 = [0.2,0.4,0.6,0.8,1,1.2]
y1 = [23.08+ 3.06,20.38+ 3.06,20.00+ 3.06,21.62+ 3.06,21.32+ 3.06,21.29+ 3.06] #tau
x_2 = [1.4,1.6,1.8,2.0,2.2,2.4] 
y2 = [21.31+ 3.06,21.26+ 3.06,20.38+ 3.06,20.84+ 3.06,20.22+ 3.06,20.23+ 3.06] #labmda

fig = plt.figure(figsize=(5.5, 4))

colors = [[59, 108, 147], [253, 187, 71], [161,37,39], [96,150,61]]
for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r/255, g/255, b/255)
color_index = [0, 1, 2, 3]
colors = [colors[i] for i in color_index]
axes = fig.add_subplot(1, 1, 1)

bwith = 1.3
axes.spines['bottom'].set_linewidth(bwith)
axes.spines['left'].set_linewidth(bwith)
axes.spines['top'].set_linewidth(bwith)
axes.spines['right'].set_linewidth(bwith)

# axes.plot(x1, y1, color=colors[0], marker='o',markersize=17, markerfacecolor=colors[0], markeredgecolor=colors[0], linewidth=3.5)
axes.plot(x_2, y2, color='#8B4513', marker='D',markersize=17, markerfacecolor='#8B4513', markeredgecolor='#8B4513', linewidth=3.5)


y_major_locator=MultipleLocator(0.7)
x_major_locator=MultipleLocator(0.2)


ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)

plt.ylabel('ACER', fontsize=28)
plt.xlabel('$\lambda$', fontsize=28)

# plt.xticks(x1, [classes for classes in x_1] ,color='black', fontsize=22)
# plt.yticks(color='w')


plt.xticks(color='black',fontsize=22)
plt.yticks(fontsize=22)
plt.ylim((22.6, 26.4))


plt.grid(color='#B1B1B1', linewidth=0.7)
# plt.legend(bbox_to_anchor=(0, 1.05), ncol=4, loc=3, borderaxespad=0,prop={'size': 23})

img_save_path = '/remote-home/mengxichen/MMANet-CVPR2023-main/classification/line_p1.pdf'
plt.savefig(img_save_path, dpi=3000, bbox_inches='tight')