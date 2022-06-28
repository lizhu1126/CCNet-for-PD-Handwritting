#use this file for processing visualization

import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_lossANDval(n):
    #fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12.8,4.8))
    
    for i in range(0,n-1):
        encVal = np.load('.yourpath/your.npy'.format(n-1))
        a=[i,i+1]
        b=[encVal[i], encVal[i+1]]
        ax1.plot(a, b, color='r', label='val')

    for j in range(0,n-1):
        encLoss = np.load('.yourpath/your.npy'.format(n-1))\
        c=[j,j+1]
        d=[encLoss[j], encLoss[j+1]]
        ax2.plot(c, d, color='r', label='loss')

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('valAccuracy')

    plt.savefig("./lossVal.jpg")


if __name__=="__main__":
    m=1500
    plot_lossANDval(m)
    #plot_val(m)






























