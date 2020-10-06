from data_generator_read_config import read_arguments_from_file
from datagenerator import DataGenerator
from csvreader import readcsv, writecsv

import matplotlib.pyplot as plt
import numpy as np

def remove_datapoints(data, time, remove_factor):
    for i in range(len(data)):
        num_elem = int(remove_factor * len(data[i]))
        indx = np.random.choice(len(data[i]), num_elem, replace=False)
        indx = np.sort(indx)[::-1]
        data[i] = np.delete(data[i], indx, axis=0)
        time[i] = np.delete(time[i], indx, axis=0)
    return data, time

def remove_parabolic_under_zero(data, y_index, time):
    for i in range(len(data)):
        if np.min(data[i][:,:,y_index]) < 0:
            indx = np.argmin(data[i][:,:,y_index] > 0) + 1
            data[i] = data[i][0:indx]
            time[i] = time[i][0:indx]
    return data, time

def test_plot_data(fp, device):
    # Read and plot the data just to look at it.
    data, _, times, _, dim, ntraj = readcsv(fp, device)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, traj in enumerate(data):
        plt.plot(traj[:,:,0].flatten(), traj[:,:,1].flatten(), '-o')#, c=colors[idx])
        
    plt.figure()
    for idx, (time, traj) in enumerate(zip(times, data)):
        plt.plot(time.flatten(), traj[:,:,0].flatten(), '--')#, c=colors[idx])
        plt.plot(time.flatten(), traj[:,:,1].flatten(), '-')#, c=colors[idx])

    '''
    plt.figure()
    for idx, (time, traj) in enumerate(zip(times, data)):
        plt.plot(time.flatten(), traj[:,:,2].flatten(), '--')#, c=colors[idx])
        plt.plot(time.flatten(), traj[:,:,3].flatten(), '-')#, c=colors[idx])

    plt.figure()
    for idx, (time, traj) in enumerate(zip(times, data)):
        plt.plot(time.flatten(), traj[:,:,4].flatten(), '--')#, c=colors[idx])
        plt.plot(time.flatten(), traj[:,:,5].flatten(), '-')#, c=colors[idx])
    '''
    plt.savefig(fp.replace("csv", "png", 1).replace("data","plots"))
    
if __name__=='__main__':
    settings = read_arguments_from_file("generator_config.cfg")

    function = settings['function']
    filename_extension = settings['filename_extension']
    filename = "{}_{}.csv".format(function, filename_extension)
    save_fp = 'data/{}'.format(filename)
    remove_factor = settings['remove_factor']
    ntraj = settings['ntraj']
    y0_range = settings['y0_range']
    t_range = settings['t_range']
    num_times = settings['num_times']
    noise_scale = settings['noise_scale']
    device = settings['device']
    random_y0 = settings['random_y0']



    if function == "1d_parabolic":
        assert len(y0_range) == 4
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
        data, time = remove_parabolic_under_zero(data, 0, time)
        
    elif function == "2d_parabolic":
        assert len(y0_range) == 8
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
        data, time = remove_parabolic_under_zero(data, 1, time)
    elif function == "2d_parabolic_drag":
        assert len(y0_range) == 8
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_sdevicecale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
        data, time = remove_parabolic_under_zero(data, 1, time)
    elif function == "simple_harmonic":
        assert len(y0_range) == 4
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
    elif function == "damped_harmonic":
        assert len(y0_range) == 4
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
    elif function == "mystery_function":
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()
    elif function == "lotka_volterra":
        dg = DataGenerator(ntraj=ntraj, y0_range=y0_range, t_range=t_range, num_times=num_times, noise_scale=noise_scale, device=device, function=function, random_y0=random_y0)
        data, _, time, _ = dg.generate()


    data, time = remove_datapoints(data, time, remove_factor)  
    writecsv(fp=save_fp, dim=int(len(y0_range)/2), ntraj=ntraj, data_np=data, t_np=time)

    #save_fp = 'D:\Skola\MSc-Thesis\Base\data\mystery_function_test.csv'
    test_plot_data(save_fp, device)