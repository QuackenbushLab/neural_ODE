from csvreader import readcsv, writecsv

import matplotlib.pyplot as plt
import numpy as np
import os


def test_plot_data(fp, device):
    # Read and plot the data just to look at it.
    data, _, times, _, dim, ntraj = readcsv(fp, device)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    '''
    for idx, traj in enumerate(data):
        plt.plot(traj[:,:,0].flatten(), traj[:,:,1].flatten(), '-o')#, c=colors[idx])
    '''     
    plt.figure()
    for idx, (this_times, traj) in enumerate(zip(times, data)):
        plt.plot(this_times.flatten(), traj[:,:,0].flatten(), '-')#, c=colors[idx])
        plt.plot(this_times.flatten(), traj[:,:,139].flatten(), '-o')#, c=colors[idx])
    save_plot = fp.replace("csv", "png", 1).replace("data","plots")
    plt.savefig(save_plot)
    
if __name__=='__main__':
    save_fp = "data/simulated_expression_chalmers_150genes_20200818.csv"
    test_plot_data(save_fp, "cpu")
