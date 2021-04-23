import os
import warnings
from datetime import datetime
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
from h5py import File
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    from joblib import Memory, delayed
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.utils.config import process_config
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    from src.utils.parallel_bar import ParallelExecutor

DEFAULT_HYP_LENGTH = 30


class MultiCohortDataset(Dataset):

    def __init__(self, config, subject=None, subset='test'):
        self.config = config
        self.subject = subject
        self.subset = subset

        self.batch_size = config.data_loader.batch_size[subset]
        self.data = config.data_loader.data[subset]
        self.data_dir = config.data_loader.data_dir
        self.train_fraction = config.data_loader.train_fraction if subset == 'train' else None
        self.fs = 128  # TODO: put into yaml
        self.modalities = config.data_loader.modalities
        self.num_channels = len(self.modalities) + 1
        self.num_classes = config.data_loader.num_classes
        self.segment_length = config.data_loader.segment_length
        assert self.segment_length % DEFAULT_HYP_LENGTH == 0, 'Segment length should be a multiple of 30 s!'

        self.df = None
        self.indexes = None
        self.length_recordings = None
        self.num_subjects = None

        # Collect dataframes for cohorts
        cohorts = [data[0] for data in self.data]
        subsets = [data[1] for data in self.data]
        df = pd.concat(
            [pd.read_csv(os.path.join(self.data_dir, 'csv', cohort + '.csv'), index_col=0) for cohort
             in list(set(cohorts))], ignore_index=True)

        # Prune for subsets in cohorts
        self.df = []
        for c, s in zip(cohorts, subsets):
            self.df.append(df.loc[(df.Cohort == c) & (df.Partition == s)])
        self.df = pd.concat(self.df, ignore_index=True)

        # Prune for subsets in files
        list_files = os.listdir(os.path.join(self.data_dir, 'h5'))
        self.df = self.df.loc[df.FileID.str.lower().isin([f[:-3] for f in list_files]), :] \
                         .sort_values(by=['Cohort', 'FileID']) \
                         .reset_index(drop=True)

        # Prune for skipped subjects
        self.df = self.df.loc[pd.isna(self.df.Skip)].reset_index(drop=True)

        # Maybe only take a fraction of training data. This routine sorts the available cohorts by size and adjusts the number of
        # PSGs taken from each successively.
        if self.train_fraction:
            print('Using {} of the data'.format(self.train_fraction))
            if self.train_fraction > 1.0:
                grab_from_each = int(self.train_fraction // len(cohorts))
                n_cohort = {c: None for c in cohorts}
                sum_in_cohorts = [(c, sum(self.df.Cohort == c)) for c in cohorts]
                total = 0
                remaining_cohorts = len(cohorts)
                for c, cohort_sum in sorted(sum_in_cohorts, key=lambda cohort_sum: cohort_sum[1]):
                    if cohort_sum >= grab_from_each:
                        n_cohort[c] = grab_from_each
                        total += grab_from_each
                        remaining_cohorts -= 1
                    else:
                        n_cohort[c] = cohort_sum
                        total += cohort_sum
                        remaining_cohorts -= 1
                        if remaining_cohorts > 0:
                            grab_from_each = int((self.train_fraction - total) // remaining_cohorts)
                        else:
                            print('[ Warning ] No more cohorts to draw data from, lower the requested amount of data!')
                print(f'[ Info ] Requested {self.train_fraction} / received {total}')
                # n_cohort = {c: np.minimum(int(self.train_fraction // len(cohorts)), sum(self.df.Cohort == c)) for c in cohorts}
            df_frac = []
            for c in cohorts:
                if self.train_fraction <= 1.0:
                    df_frac.append(self.df.loc[self.df.Cohort == c].sample(frac=self.train_fraction))
                else:
                    df_frac.append(self.df.loc[self.df.Cohort == c].sample(n=n_cohort[c]))
            self.df = pd.concat(df_frac, ignore_index=True).sort_values(by=['Cohort', 'FileID']) \
                                                           .reset_index(drop=True)

        # Maybe select single subject
        if self.subject:
            self.df = self.df[self.df.FileID == self.subject].sort_values(by=['Cohort', 'FileID']) \
                                                             .reset_index(drop=True)

        self.num_subjects = len(self.df)
        print('Number of subjects: {}'.format(self.num_subjects))

        # We change the hypnogram length depending on the segment size
        self.mult_factor = self.segment_length // DEFAULT_HYP_LENGTH
        self.length_recordings = (
            self.df.HypnogramLength.values // self.mult_factor).astype(np.int)
        self.df['Length'] = self.length_recordings
        self.df['FileLength'] = self.df.HypnogramLength * 30 * self.fs

        self.indexes = [(fid, j * self.mult_factor + range(self.mult_factor)) for i, fid in zip(np.arange(self.num_subjects), self.df['FileID']) for j in np.arange(self.length_recordings[i])]

        def get_h5(file):
            with File(os.path.join(self.data_dir, 'h5', file.lower()), 'r') as db:
                with np.printoptions(precision=2, threshold=5, edgeitems=1):
                    hypnogram = db['hypnogram'][:].astype(np.uint8)
                    psg = db['data'][:].astype(np.float32)

            return file.split('.')[0], psg, hypnogram

        # Preloading data as mmaps with joblib
        self.cache_dir = 'data/processed/.cache'
        memory = Memory(self.cache_dir, mmap_mode='r', verbose=0)
        get_data = memory.cache(get_h5)

        self.data = {r: None for r in self.df.FileID}
        self.hypnogram = {r: None for r in self.df.FileID}

        with np.printoptions(precision=2, threshold=5, edgeitems=1):
            data = ParallelExecutor(n_jobs=-1, prefer="threads")(total=len(self.df))(
                delayed(get_data)(k + '.h5') for k in self.data.keys()
            )
        for record, psg, hypnogram in tqdm(data, desc='Processing... '):
            self.data[record] = psg
            self.hypnogram[record] = hypnogram

    def __len__(self):
        return sum(self.length_recordings)

    def __getitem__(self, idx):
        file_id, position = self.indexes[idx]
        position = self.indexes[idx][1]
        cohort = self.df.loc[self.df.FileID == file_id, 'Cohort'].values[0]
        data = self.data[file_id][:, position, :]
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
    num_workers = 0
    print('Num workers: {}'.format(num_workers))
    config = process_config('./src/configs/exp03-frac100.yaml')
    s = datetime.now()
    train_data = MultiCohortDataset(config, subset='train')
    data = next(iter(train_data))
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
