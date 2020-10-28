#from datagenerator import DataGenerator
from csvreader import readcsv, writecsv
import numpy as np
import torch
from math import ceil
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

class DataHandler:

    def __init__(self, data_np, data_pt, time_np, time_pt, dim, ntraj, val_split, device, normalize, batch_type, batch_time, batch_time_frac, data_np_0noise, data_pt_0noise):
        self.data_np = data_np
        self.data_pt = data_pt
        self.data_np_0noise = data_np_0noise
        self.data_pt_0noise = data_pt_0noise
        self.time_np = time_np
        self.time_pt = time_pt
        self.dim = dim
        self.ntraj = ntraj
        self.batch_type = batch_type
        self.batch_time = batch_time
        self.batch_npoints = int(ceil(batch_time_frac * batch_time))
        if normalize:
            self._normalize()
        self.device = device
        self.val_split = val_split
        self.epoch_done = False
        #self.noise = noise

        self._calc_datasize()
        if batch_type == 'single':
            self._split_data_single(val_split)
            self._create_validation_set_single()
        elif batch_type == 'trajectory':
            self._split_data_traj(val_split)
            self._create_validation_set_traj()
        elif batch_type == 'batch_time':
            self._split_data_time(val_split)
            self._create_validation_set_time()
        else:
            print("Invalid batch type: '{}'".format(batch_type))
            raise ValueError

    @classmethod
    def fromcsv(cls, fp, device, val_split, normalize=False, batch_type='single', batch_time=1, batch_time_frac=1.0, noise = 0):
        ''' Create a datahandler from a CSV file '''
        data_np, data_pt, t_np, t_pt, dim, ntraj, data_np_0noise, data_pt_0noise = readcsv(fp, device, noise_to_add = noise)
        return DataHandler(data_np, data_pt, t_np, t_pt, dim, ntraj, val_split, device, normalize, batch_type, batch_time, batch_time_frac, data_np_0noise, data_pt_0noise)

    @classmethod
    def fromgenerator(cls, generator, val_split, device, normalize=False):
        ''' Create a datahandler from a data generator '''
        data_np, data_pt, t_np, t_pt = generator.generate()
        return DataHandler(data_np, data_pt, t_np, t_pt, generator.dim, generator.ntraj, val_split, device, normalize)

    def saveascsv(self, fp):
        ''' Saves the data to a CSV file '''
        writecsv(fp, self.dim, self.ntraj, self.data_np, self.time_np)

    def _normalize(self):
        max_val = 0
        for data in self.data_pt:
            if torch.max(torch.abs(data)) > max_val:
                max_val = torch.max(torch.abs(data))
        for i in range(self.ntraj):
            self.data_pt[i] = torch.div(self.data_pt[i], max_val)
            self.data_np[i] = self.data_np[i] / max_val.numpy()

    def reset_epoch(self):
        self.train_set = self.train_set_original.copy()
        self.epoch_done = False

    def get_batch(self, batch_size):
        if self.batch_type == 'single':
            return self._get_batch_single(batch_size)
        elif self.batch_type == 'trajectory':
            return self._get_batch_traj(batch_size)
        elif self.batch_type == 'batch_time':
            return self._get_batch_time(batch_size)

    def _get_batch_single(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > batch_size:
            indx = np.random.choice(train_set_size, batch_size, replace=False)
        else:
            indx = np.arange(train_set_size)
            self.epoch_done = True # We are doing the last items in current epoch
        indx = np.sort(indx)[::-1]
        batch_indx = [self.train_set[x] for x in indx]
        batch = []
        t = []
        target = []
        for i in batch_indx:
            batch.append(self.data_pt[i[0]][i[1]])
            target.append(self.data_pt[i[0]][i[1] + 1])
            t.append(torch.stack([self.time_pt[i[0]][i[1] + ii] for ii in range(2)]))
        for i in indx:
            self.train_set.pop(i)
        # Convert the lists to tensors
        reshape_size = batch_size if train_set_size > batch_size else train_set_size
        batch = torch.stack(batch).to(self.device)
        t = torch.stack(t).to(self.device)
        target = torch.stack(target).to(self.device)
        return batch, t, target

    def _get_batch_traj(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > 1:
            i = np.random.choice(train_set_size, replace=False)
            indx = self.train_set[i]
            self.train_set = np.delete(self.train_set, i)
        else:
            indx = self.train_set[0]
            self.epoch_done = True # We are doing the last items in current epoch
        # Convert the lists to tensors
        batch = self.data_pt[indx][0:-1].to(self.device)
        t = []
        for i in range(self.time_pt[indx].shape[0] - 1):
            t.append(torch.tensor([self.time_pt[indx][i], self.time_pt[indx][i+1]]))
        t = torch.stack(t)
        target = self.data_pt[indx][1::].to(self.device)
        return batch, t, target

    def _get_batch_time(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > batch_size:
            indx = np.random.choice(train_set_size, batch_size, replace=False)
        else:
            indx = np.arange(train_set_size)
            self.epoch_done = True # We are doing the last items in current epoch
        indx = np.sort(indx)[::-1]
        batch_indx = [self.train_set[x] for x in indx]
        batch = []
        t = []
        target = []
        for i in batch_indx:
            sub_indx = np.random.choice(np.arange(start=i[1], stop=i[1]+self.batch_time), size=self.batch_npoints, replace=False)
            sub_indx = np.sort(sub_indx)
            sub_indx = np.append(sub_indx, sub_indx[-1]+1)
            batch.append(torch.stack([self.data_pt[i[0]][ii] for ii in sub_indx[0:-1]]))
            target.append(torch.stack([self.data_pt[i[0]][ii+1] for ii in sub_indx[1::]]))
            t.append(torch.tensor([self.time_pt[i[0]][ii] for ii in sub_indx]))
        for i in indx:
            self.train_set.pop(i)
        # Convert the lists to tensors
        reshape_size = batch_size if train_set_size > batch_size else train_set_size
        batch = torch.stack(batch).squeeze().to(self.device)
        t = torch.stack(t).to(self.device)
        target = torch.stack(target).squeeze().to(self.device)
        return batch, t, target

    def _split_data_single(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int((self.datasize - self.ntraj) * val_split)
        all_indx = np.arange(len(self.indx))
        val_indx = np.random.choice(all_indx, size=self.n_val, replace=False)
        train_indx = np.setdiff1d(all_indx, val_indx, assume_unique=True)
        self.val_set_indx = [self.indx[x] for x in val_indx]
        self.train_set_original = [self.indx[x] for x in train_indx]
        self.train_data_length = len(self.train_set_original)
      

    def _split_data_traj(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int(round(self.ntraj * val_split))
        self.val_set_indx = np.random.choice(np.arange(self.ntraj), size=self.n_val, replace=False)
        traj_indx = np.arange(self.ntraj)
        self.train_set_original = np.setdiff1d(traj_indx, self.val_set_indx)
        self.train_data_length = len(self.train_set_original)

    def _split_data_time(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int((self.datasize - self.ntraj) * val_split)
        all_indx = np.arange(len(self.indx))
        val_indx = np.random.choice(all_indx, size=self.n_val, replace=False)
        train_indx = np.setdiff1d(all_indx, val_indx, assume_unique=True)
        self.val_set_indx = [self.indx[x] for x in val_indx]
        self.train_set_original = [self.indx[x] for x in train_indx]
        self.train_data_length = len(self.train_set_original)
        
    def _calc_datasize(self):
        if self.batch_type == 'batch_time':
            self._calc_datasize_time()
        else:
            self._calc_datasize_single()

    def _calc_datasize_single(self):
        self.datasize = 0
        self.indx = []
        row_indx = 0
        for row in self.time_np:
            rowsize = row.size
            self.datasize += rowsize
            indices = np.arange(rowsize - 1)
            indices = [(row_indx, x) for x in indices]
            self.indx.extend(indices)
            row_indx += 1

    def _calc_datasize_time(self):
        self.datasize = 0
        self.indx = []
        row_indx = 0
        for row in self.time_np:
            rowsize = row.size - self.batch_time
            self.datasize += rowsize
            indices = np.arange(rowsize - 1)
            indices = [(row_indx, x) for x in indices]
            self.indx.extend(indices)
            row_indx += 1

    #def get_y0(self):
    #    return [tensor[0] for tensor in self.data_pt]
    
    def get_mu0(self):
        return [tensor[0] for tensor in self.data_pt_0noise]

    def get_true_mu_set(self):
        all_indx = [self.indx[x] for x in np.arange(len(self.indx))]
        mean_data = []
        mean_target = []
        mean_t = []
        for i in all_indx:
            mean_data.append(self.data_pt_0noise[i[0]][i[1]])
            mean_target.append(self.data_pt_0noise[i[0]][i[1] + 1])
            mean_t.append(torch.stack([self.time_pt[i[0]][i[1] + ii] for ii in range(2)]))
        mean_data = torch.stack(mean_data).to(self.device)
        mean_target = torch.stack(mean_target).to(self.device)
        mean_t = torch.stack(mean_t).to(self.device)
        return mean_data, mean_t, mean_target
       
    def get_times(self):
        times = torch.stack(self.time_pt)
        return times

    def calculate_trajectory(self, odenet, method, num_trajs):
        trajectories = []
        mu0 = self.get_mu0()
        for j in range(num_trajs):
            if odenet.explicit_time:
                _y = torch.cat((mu0[j], self.time_pt[j][0].reshape((1, 1))), 1)
            else:
                _y = mu0[j]
            y = odeint(odenet, _y, self.time_pt[j], method=method)
            y = torch.Tensor.cpu(y)
            trajectories.append(y)
        return trajectories

    def _create_validation_set_single(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_target = []
        self.val_t = []
        if self.val_set_indx:
            for i in self.val_set_indx:
                self.val_data.append(self.data_pt[i[0]][i[1]])
                self.val_target.append(self.data_pt[i[0]][i[1] + 1])
                self.val_t.append(torch.stack([self.time_pt[i[0]][i[1] + ii] for ii in range(2)]))
            self.val_data = torch.stack(self.val_data).to(self.device)
            self.val_target = torch.stack(self.val_target).to(self.device)
            self.val_t = torch.stack(self.val_t).to(self.device)

    def _create_validation_set_traj(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_target = []
        self.val_t = []
        if self.val_set_indx.any():
            for i in self.val_set_indx:
                self.val_data.append(self.data_pt[i][0:-1]) 
                self.val_target.append(self.data_pt[i][1::])
                self.val_t.append(self.time_pt[i])
            self.val_data = torch.stack(self.val_data).to(self.device) #IH addition
            self.val_target = torch.stack(self.val_target).to(self.device) #IH addition
            self.val_t = torch.stack(self.val_t).to(self.device) #IH addition

    def _create_validation_set_time(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_target = []
        self.val_t = []
        if self.val_set_indx:
            for i in self.val_set_indx:
                self.val_data.append(self.data_pt[i[0]][i[1]:i[1]+self.batch_time])
                self.val_target.append(self.data_pt[i[0]][i[1]+1:i[1]+self.batch_time+1])
                self.val_t.append(torch.tensor([self.time_pt[i[0]][i[1] + ii] for ii in range(self.batch_time+1)]))
            self.val_data = torch.stack(self.val_data).squeeze().to(self.device)
            self.val_target = torch.stack(self.val_target).squeeze().to(self.device)
            self.val_t = torch.stack(self.val_t).to(self.device)

    def get_validation_set(self):
        return self.val_data, self.val_t, self.val_target, self.n_val