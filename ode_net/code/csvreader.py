import torch
import numpy as np
import csv
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

def readcsv(fp, device):
    print("Reading from file {}".format(fp))
    data_np = []
    data_pt = []
    t_np = []
    t_pt = []
    with open(fp, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = []
        for r in reader:
            rows.append(r)
        dim = int(rows[0][0])
        ntraj = int(rows[0][1])
        data = rows[1:]
        for traj in range(ntraj):
            current_length = len(data[traj*(dim+1)])
            traj_data = np.zeros((current_length, 1, dim), dtype=np.float32)
            for d in range(dim + 1):
                row = [float(f) for f in data[traj*(dim+1) + d]]
                if d == dim:
                    # Row is time data
                    t_np.append(np.array(row))
                    t_pt.append(torch.tensor(row).to(device))
                else:
                    traj_data[:,:,d] = np.expand_dims(np.array(row), axis=1)
            data_np.append(traj_data)
            data_pt.append(torch.tensor(traj_data).to(device))
    return data_np, data_pt, t_np, t_pt, dim, ntraj

def writecsv(fp, dim, ntraj, data_np, t_np):
    ''' Write data from a datagenerator to a file '''
    # Clear the file
    f = open(fp, "w+")
    f.close()

    # Write the data into the file
    info = np.array([dim, ntraj])
    with open(fp, 'a') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(info)
        for i in range(ntraj):
            for j in range(dim):
                writer.writerow(data_np[i][:,:,j].flatten())
            writer.writerow(t_np[i])
    print("Written to file {}".format(fp))
