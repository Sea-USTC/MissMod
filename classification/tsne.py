# from openTSNE import TSNE
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator


RS=2020
# sns.set_palette('muted') 
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def scatter(x, labels):
    # log = ((log.max()-log)/(log.max()-log.min()))*2 -1
    # log = 1 / (1 + np.exp(-log))
    x_min, x_max = x.min(0), x.max(0)
    # x = (x - x_min) / (x_max - x_min)  # 归一化
    data = x
    f = plt.figure()
    ax = plt.subplot(aspect='equal')

    colors = [[59, 108, 147], [253, 187, 71], [161,37,39], [96,150,61]]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r/255, g/255, b/255)
    color_index = [0, 1, 2, 3]
    colors = [colors[i] for i in color_index]
    positions_0 = np.where(labels==0)

    sc = ax.scatter(data[positions_0[0], 0], data[positions_0[0], 1], lw=0, s=4.0, c=colors[2])#, cmap=plt.cm.Blues, label='{}'.format(1))
    
    positions_1 = np.where(labels==1)
    sc = ax.scatter(data[positions_1[0], 0], data[positions_1[0], 1], lw=0, s=4.0, c=colors[3])#, cmap=plt.cm.RdPu, label='{}'.format(1))

    
    # sc = ax.scatter(x[-4:,0], x[-4:,1], lw=0, s=5.0)
    ax.axis('off')
    ax.axis('tight')

    txts = []

    return f, ax, sc, txts


if __name__ == '__main__':
    colors = [[59, 108, 147], [253, 187, 71], [161,37,39], [96,150,61]]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r/255, g/255, b/255)
    color_index = [0, 1, 2, 3]
    colors = [colors[i] for i in color_index]

    data_100 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/100.npy")
    data_010 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/010.npy")
    data_001 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/001.npy")
    data_110 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/110.npy")
    data_101 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/101.npy")
    data_011 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/011.npy")
    data_111 = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/111.npy")
    label = np.load("/remote-home/mengxichen/MMANet-CVPR2023-main/classification/label.npy")

    data = np.concatenate([data_100,data_010,data_001,data_110,data_101,data_011,data_111],axis=0)
    data = np.exp(data)/np.expand_dims(np.sum(np.exp(data),axis=-1),axis=1)
    labels = np.concatenate([label,label,label,label,label,label,label],axis=0)


    logits = data[range(0,data.shape[0]),labels]

    index = np.bitwise_and(logits>0.3,logits<0.7)
    logits = logits[index]
    labels = labels[index]

    positions_0 = np.where(labels==0)
    positions_1 = np.where(labels==1)
    data_0 = logits[positions_0]*100
    data_1 = (1-logits[positions_1])*100

    score0 = pd.Series(data_0)
    score1 = pd.Series(data_1)
    # se1 = pd.cut(score1, bins=range(0,101), labels=range(0,100))
    # se0 = pd.cut(score0, bins=range(0,101), labels=range(0,100))


    se1 = pd.cut(score1, bins=range(30,71), labels=range(30,70))
    se0 = pd.cut(score0, bins=range(30,71), labels=range(30,70))

    bin_counts0 = se0.value_counts()
    bin_counts1 = se1.value_counts()

    # 对频数取对数
    log_bin_counts0 = bin_counts0**(1/2)
    log_bin_counts1 = bin_counts1**(1/2)


    fig = plt.figure(figsize=(5.5,6))
    plt.grid(color='#B1B1B1', linewidth=0.7)
    _, axes = plt.subplots()
    # bwith = 1.7
    bwith = 2
    axes.spines['bottom'].set_linewidth(bwith)
    axes.spines['left'].set_linewidth(bwith)
    axes.spines['top'].set_linewidth(bwith)
    axes.spines['right'].set_linewidth(bwith)

    # y_major_locator=MultipleLocator(60)
    y_major_locator=MultipleLocator(10)

    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    



    axes.bar(log_bin_counts1.index,log_bin_counts1, alpha=0.8, color='#CD853F')
    axes.bar(log_bin_counts0.index,log_bin_counts0, alpha=0.6, color='#20B2AA')
    
    # plt.legend(['Class 0','Class 1'],prop={'size': 17},loc='upper center',ncol=2)

    plt.grid(axis='y', color='#B1B1B1', linewidth=0.7, linestyle='--')

    # plt.ylim((0, 350))

    plt.ylim((0, 33))

    # plt.ylabel('Sample Num', fontsize=28)
    # plt.xlabel('Logits ($c=0$)', fontsize=28)
    # plt.xticks(color='black',fontsize=22)
    # plt.xticks([0,20,40,60,80,100], [0.0,0.2,0.4,0.6,0.8,1.0] ,color='black',fontsize=22)
    plt.xticks([30,40,50,60,70], [0.3,0.4,0.5,0.6,0.7] ,color='black',fontsize=30)
    plt.yticks(color='black',fontsize=30)#22
    # plt.yticks(color='w')
    path = "/remote-home/mengxichen/MMANet-CVPR2023-main"
    img_save_path = os.path.join(path, 'tsne(D).pdf')

    plt.savefig(img_save_path, dpi=1200, bbox_inches='tight')

