import os
from datetime import datetime
from joblib import Parallel, delayed

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import pdb

from src.utils.config import process_config
from src.data_loader.dataset import MultiCohortDataset
from src.utils.parallel_bar import ParallelExecutor

DEFAULT_HYP_LENGTH = 30


class BalancedDataset(MultiCohortDataset):

    def __init__(self, config, subject=None, subset='test'):
        super().__init__(config, subject=subject, subset=subset)

    def __len__(self): # Todo: remake
        return sum(self.length_recordings)

    def __getitem__(self, idx): # Todo: remake

        file_id = self.df.FileID[self.indexes[idx][0]]
        position = self.indexes[idx][1]
        cohort = self.df.loc[self.df.FileID == file_id, 'Cohort'].values[0]
        with h5py.File(os.path.join(self.data_dir, 'h5_single', '{}.h5'.format(file_id.lower())), 'r') as db:
            data = db['data'][:, position, :]
        hypnogram = self.hypnogram[file_id][position]
        out = {'fid': file_id,
               'position': position,
               'data': torch.from_numpy(data.reshape((self.num_channels, -1))[np.newaxis, :, :]),
               'target': torch.LongTensor(np.repeat(hypnogram, DEFAULT_HYP_LENGTH))}
        return out


    def get_subjects(self):
        return [fid for fid in self.df.FileID.values]

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_context('paper')
    num_workers = 128
    print('Num workers: {}'.format(num_workers))
    config = process_config('./src/configs/exp02-frac075.yaml')
    s = datetime.now()
    train_data = MultiCohortDataset(config, subset='train')
    e = datetime.now()
    print('{}'.format(e - s))
    eval_data = MultiCohortDataset(config, subset='eval')
    test_data = MultiCohortDataset(config, subset='test')

    train_loader = DataLoader(train_data, batch_size=config.data_loader.batch_size.train,
                              shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    # eval_loader = DataLoader(eval_data, batch_size=config.data_loader.batch_size.eval,
    #                          shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=config.data_loader.batch_size.test,
    #                          shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
    # # Create a dict with classes as keys and indices for values
    # self.sampling_dict = None
    # if self.mult_factor == 1:
    #     self.sampling_dict = {k: [
    #         idx for idx in self.indexes if self.labels[self.df.FileID[idx[0]]][idx[1]] == k] for k in range(self.num_classes)}
    num_epochs = 5
    start_time = datetime.now()
    for n in range(num_epochs):
        print('\nEpoch {} of {}'.format(n+1, num_epochs))
        # for idx, batch in tqdm(enumerate(train_data), total=len(train_data)):
        #     pass
        # for idx, batch in tqdm(enumerate(eval_data), total=len(eval_data)):
        #     pass
        # for idx, batch in tqdm(enumerate(test_data), total=len(test_data)):
        #     pass
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            pass
        print(idx, batch[0].size(), batch[1].size())
        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            pass
    end_time = datetime.now()
    print('\nElapsed time: {} | Time per epoch: {}'.format(end_time - start_time, (end_time - start_time)/num_epochs))
    # for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    #     pass


    # plt.plot(batch[0][np.random.randint(
    #     0, train_data.batch_size-1), 0, :, :].numpy().T + 2*np.arange(4))
    # plt.show()
