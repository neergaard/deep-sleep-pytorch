from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
from argparse import ArgumentParser
from glob import glob
from random import seed, shuffle
from datetime import datetime
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import pyedflib
from scipy import signal

from src.utils.parseXmlEdfp import parse_hypnogram
from src.utils.segmentation import segmentPSG

# Load configuration file
parser = ArgumentParser()
parser.add_argument(
    '-c', '--config-file',
    dest='config',
    type=str,
    default='data_mros.json',
    help='Configuration JSON file.'
)

args = parser.parse_args()
with open(os.path.join('./src/configs', args.config), 'r') as f:
    config = json.load(f)

# Define the cohorts
COHORTS = config['COHORTS']
COHORT_OVERVIEW_FILE = config['COHORT_OVERVIEW_FILE']
OUTPUT_DIRECTORY = config['OUTPUT_DIRECTORY']
SUBSETS = ['train', 'eval', 'test']
FILTERS = config['FILTERS']
SEGMENTATION = config['SEGMENTATION']
PARTITIONS = config['PARTITIONS']

# Define a logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# Create folder(s) if not available
if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, 'csv')):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'csv'))
if not os.path.exists(os.path.join(OUTPUT_DIRECTORY, 'h5')):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'h5'))


# Create the filters
def createPSGfilters(config_filters):
    channels = ['eeg', 'eog', 'emg']
    sos = {key: [] for key in channels}
    fs = config_filters['fs_resampling']
    order = config_filters['order']
    fc = config_filters['fc']
    btype = config_filters['btype']

    for channel in channels:
        N = order[channel]
        Wn = [2 * f / fs for f in fc[channel]]
        sos[channel] = signal.butter(
            order[channel], Wn, btype[channel], output='sos')

    return sos


sos = createPSGfilters(FILTERS)


def write_H5(psg, hypnogram, N, series, subset=None):
    filename = os.path.join(OUTPUT_DIRECTORY, 'h5', series.FileID.lower() + '.h5')
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('data', data=psg)
        dset = f.create_dataset('hypnogram', data=hypnogram)


def process_file(series_file):
    fileID = series_file['FileID']
    file_edf = series_file['File']
    file_hypnogram = series_file['Hypnogram']
    cohort = series_file['Cohort']
    subset = series_file['Partition']
    base_dir = os.path.join(file_edf[:file_edf.find(cohort)], cohort)

    # We skip the file if the hypnogram and fileID do not match up
    skip_file = False
    if cohort == 'shhs' or cohort == 'mros':
        if fileID != os.path.split(file_hypnogram)[1][:-9]:
            skip_file = True
    elif cohort == 'wsc' or cohort == 'ssc':
        if fileID != os.path.split(file_hypnogram)[1][:-4]:
            skip_file = True
    if skip_file:
        LOG.info(
            '{: <5} | {: <5} | {: <5} | No matching hypnogram file'.format(
                cohort, subset, fileID, file_hypnogram))
        return None, None

    # Load the JSON file containing channel labels
    signal_labels_json_path = os.path.join(
        './src/configs/signal_labels', '{}.json'.format(cohort))
    # if not os.path.exists(signal_labels_json_path):
    #     channel_label_identifier.run(
    #         [base_dir, 'C3', 'C4', 'A1', 'A2', 'EOGL', 'EOGR', 'EMG', 'LChin', 'RChin'])
    with open(signal_labels_json_path, 'r') as f:
        cohort_labels = json.load(f)

    # Load hypnogram and change
    try:
        LOG.info(
            '{: <5} | {: <5} | {: <5} | Loading matching hypnogram'.format(
                cohort, subset, fileID))
        if cohort in ['mros', 'shhs', 'mesa', 'cfs']:
            df_hypnogram = parse_hypnogram(file_hypnogram, 'xml')
            hypnogram = df_hypnogram['label'].values
        elif cohort == 'wsc' or cohort == 'ssc':
            hypnogram = []
            try:
                with open(file_hypnogram, 'r') as hyp_file:
                    for line in csv.reader(hyp_file, delimiter='\t'):
                        hypnogram.append(int(line[1]))
            except:
                with open(file_hypnogram, 'r') as hyp_file:
                    for line in csv.reader(hyp_file, delimiter='\t'):
                        hypnogram.append(int(float(line[0].split()[1])))
        elif cohort in ['isruc']:
            with open(file_hypnogram, 'r') as hyp_file:
                hypnogram = hyp_file.read()
                hypnogram = [int(h) for h in hypnogram.split('\n') if h]
        else:
            hypnogram = []
    except:
        return None, None
    hypnogram = np.asarray(hypnogram)

    # Figure out which channels to load
    edf = pyedflib.EdfReader(file_edf)
    n_signals = edf.signals_in_file
    sampling_frequencies = edf.getSampleFrequencies()
    signal_labels = edf.getSignalLabels()
    signal_label_idx = {category: []
                        for category in cohort_labels['categories']}
    signal_data = {category: [] for category in cohort_labels['categories']}
    rereference_data = False
    for idx, label in enumerate(signal_labels):
        for category in cohort_labels['categories']:
            if label in cohort_labels[category]:
                signal_label_idx[category] = idx
                if category in ['A1', 'A2', 'LChin', 'RChin']:
                    rereference_data = True
            else:
                continue

    # Abort if any channels are missing
    if not rereference_data:
        if any([isinstance(v, list) for k, v in signal_label_idx.items() if
                k in (['C3', 'EOGL', 'EOGR', 'EMG'])]) and any(
                [isinstance(v, list) for k, v in signal_label_idx.items() if k in (['C4', 'EOGL', 'EOGR', 'EMG'])]):
            return None, None
    else:
        if any([isinstance(v, list) for k, v in signal_label_idx.items() if k in ['A1', 'A2']]):
            return None, None

    # Load all the relevant data
    for chn, idx in signal_label_idx.items():
        if isinstance(idx, list):
            continue
        else:
            signal_data[chn] = np.zeros((1, edf.getNSamples()[idx]))
            signal_data[chn][0, :] = edf.readSignal(idx)

    # Possibly do referencing and delete ref channels afterwards
    if rereference_data:
        LOG.info(
            '{: <5} | {: <5} | {: <5} | Referencing data channels'.format(
                cohort, subset, fileID))
        left_channels = ['C3', 'EOGL']
        if signal_label_idx['A2']:
            for chn in left_channels:
                # LOG.info('Referencing {} to A2'.format(chn))
                signal_data[chn] -= signal_data['A2']
        right_channels = ['C4', 'EOGR']
        if signal_label_idx['A1']:
            for chn in right_channels:
                # LOG.info('Referencing {} to A1'.format(chn))
                signal_data[chn] -= signal_data['A1']
        if not signal_label_idx['EMG']:
            # LOG.info('Referencing LChin to RChin'.format())
            signal_data['EMG'] = signal_data['LChin'] - signal_data['RChin']
            signal_label_idx['EMG'] = signal_label_idx['LChin']
    del signal_data['A1'], signal_data['A2'], signal_data['LChin'], signal_data['RChin']

    # Resample signals
    fs = config['FILTERS']['fs_resampling']
    LOG.info('{: <5} | {: <5} | {: <5} | Resampling data'.format(
        cohort, subset, fileID))
    for chn in signal_data.keys():
        if not isinstance(signal_data[chn], list):
            signal_data[chn] = signal.resample_poly(signal_data[chn], fs, sampling_frequencies[signal_label_idx[chn]],
                                                    axis=1)

    # Decide on which EEG channel to use
    if isinstance(signal_data['C3'], list) and not isinstance(signal_data['C4'], list):
        LOG.info(
            '{: <5} | {: <5} | {: <5} | C4 is only EEG'.format(
                cohort,
                subset,
                fileID))
        eeg = signal_data['C4'].astype(dtype=np.float32)
    elif isinstance(signal_data['C4'], list) and not isinstance(signal_data['C3'], list):
        LOG.info(
            '{: <5} | {: <5} | {: <5} | C3 is only EEG'.format(cohort,
                                                               subset,
                                                               fileID))
        eeg = signal_data['C3'].astype(dtype=np.float32)
    elif not isinstance(signal_data['C3'], list) and not isinstance(signal_data['C4'], list):
        energy = [np.sum(np.abs(signal_data[chn]) ** 2)
                  for chn in ['C3', 'C4']]
        lowest_energy_channel = ['C3', 'C4'][np.argmin(energy)]
        eeg = signal_data[lowest_energy_channel].astype(dtype=np.float32)
        LOG.info('{: <5} | {: <5} | {: <5} | Selecting {} as EEG'.format(
            cohort, subset, fileID, lowest_energy_channel))
    else:
        LOG.info('Current cohort: {: <5} | Current subset: {: <5} | Current file: {: <5} | Available channels {}'.format(
            cohort, subset, fileID, [*signal_labels]))
        return None, None

    psg = {'eeg': eeg,
           'eog': np.concatenate((signal_data['EOGL'], signal_data['EOGR'])).astype(dtype=np.float32),
           'emg': signal_data['EMG'].astype(dtype=np.float32)}

    # Perform filtering
    for chn in psg.keys():
        for k in range(psg[chn].shape[0]):
            psg[chn][k, :] = signal.sosfiltfilt(sos[chn], psg[chn][k, :])

    # Do recording standardization
    for chn in psg.keys():
        for k in range(psg[chn].shape[0]):
            X = psg[chn][np.newaxis, k, :]
            m = np.mean(X)
            s = np.std(X)
            psg[chn][k, :] = (X - m)/s

    # Segment the PSG data
    psg_seg = segmentPSG(SEGMENTATION, fs, psg)

    # Also, if the signals and hypnogram are of different length, we assume that the start time is fixed for both,
    # so we trim the ends
    trim_length = np.min([len(hypnogram), psg_seg['eeg'].shape[1]])
    max_length = np.max([len(hypnogram), psg_seg['eeg'].shape[1]])
    LOG.info('{: <5} | {: <5} | {: <5} | Trim/max length: {}/{}'.format(
            cohort, subset, fileID, trim_length, max_length))
    hypnogram = hypnogram[:trim_length]
    psg_seg = {chn: sig[:, :trim_length, :] for chn, sig in psg_seg.items()}

    # Lights off/on period only
    if cohort in ['mros']:
        visit = fileID.split('-')[1]
        df = pd.read_csv('./data/raw/mros/datasets/mros-{}-dataset-0.3.0.csv'.format(visit), usecols=['nsrrid', 'poststtp', 'postlotp'])
        lights_off = datetime.strptime(df.loc[df.nsrrid.str.lower() == series_file.SubjectID, 'postlotp'].tolist()[0], '%H:%M:%S')
        study_start = datetime.strptime(df.loc[df.nsrrid.str.lower() == series_file.SubjectID, 'poststtp'].tolist()[0], '%H:%M:%S')
        if (lights_off - study_start).days == -1:
            lights_off_epoch = ((lights_off + timedelta(days=1) - study_start)/30).seconds
        else:
            lights_off_epoch = ((lights_off - study_start)/30).seconds
        LOG.info('{: <5} | {: <5} | {: <5} | Lights off at epoch {}/{}'.format(
            cohort, subset, fileID, lights_off_epoch, trim_length))
        hypnogram = hypnogram[lights_off_epoch:]
        psg_seg = {chn: sig[:, lights_off_epoch:, :] for chn, sig in psg_seg.items()}

    # We should remove hypnogram episodes which do not conform to standards, ie. (W, N1, N2, N3, R) -> (0, 1, 2, 3, 4)
    keep_idx = []
    if cohort in ['wsc', 'ssc', 'isruc']:
        hypnogram[hypnogram == 4] = 3
        hypnogram[hypnogram == 5] = 4
        keep_idx = (hypnogram <= 4) & (hypnogram >= 0)
    elif cohort in ['isruc']:
        hypnogram[hypnogram == 5] = 4
        keep_idx = hypnogram != 7
    if not isinstance(keep_idx, list):
        psg_seg = {chn: signal[:, keep_idx, :] for chn, signal in psg_seg.items()}
        hypnogram = hypnogram[keep_idx]

    return psg_seg, hypnogram


def process_cohort(paths_cohort, name_cohort):

    # Get a sorted list of all the EDFs
    if name_cohort in ['ssc']:
        list_edf = sorted(glob(paths_cohort['edf'] + '/*.[Ee][Dd][Ff]'))
    else:
        list_edf = sorted(
            glob(paths_cohort['edf'] + '/**/*.[EeRr][DdEe][FfCc]', recursive=True))

    if not list_edf:
        LOG.info('{: <5} | Cohort is empty, skipping'.format(name_cohort))
        return None

    # This returns a file ID (ie. xxx.edf becomes xxx)
    if name_cohort in ['isruc']:
        baseDir = [os.path.split(edf[:edf.find('subgroup')])[0]
                   for edf in list_edf]
        list_fileID = [fid[fid.find('subgroup'):-4] for fid in list_edf]
    else:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    # Get a list of the hypnograms
    if name_cohort in ['shhs', 'mros', 'mesa', 'cfs']:
        list_hypnogram = sorted(
            glob(paths_cohort['stage'] + '/**/*.[Xx][Mm][Ll]', recursive=True))
        list_hypnogram = [
            hyp for hyp in list_hypnogram if not hyp[-13:] == 'profusion.xml']
    elif name_cohort in ['wsc', 'ssc']:
        list_hypnogram = sorted(
            glob(paths_cohort['stage'] + '/*.[Ss][Tt][Aa]'))
    elif name_cohort in ['isruc']:
        list_hypnogram = sorted(
            glob(paths_cohort['stage'] + '/**/*_1.[Tt][Xx][Tt]', recursive=True))
    else:
        return None
    list_hypnogram = list_hypnogram[:10]

    # Make sure that we only keep those recordings who have a corresponding hypnogram
    if name_cohort == 'wsc' or name_cohort == 'ssc':
        hyp_IDs = [os.path.split(hypID)[1][:-4] for hypID in list_hypnogram]
    elif name_cohort in ['mros', 'mesa', 'shhs', 'cfs']:
        hyp_IDs = [os.path.split(hypID)[
            1][:-9] for hypID in list_hypnogram if not hypID[-13:] == 'profusion.xml']
    elif name_cohort == 'isruc':
        hyp_IDs = [hypID[hypID.find('subgroup'):-6]
                   for hypID in list_hypnogram]
    list_ID_union = list(set(list_fileID) & set(hyp_IDs))
    for id in hyp_IDs:
        if not id in list_ID_union:
            LOG.info('{: <5} | Removing {}'.format(name_cohort, id))
            list_hypnogram.remove(id)
    for id in list_fileID:
        if not id in list_ID_union:
            LOG.info('{: <5} | Removing {}'.format(name_cohort, id))
            list_edf.remove(
                list_edf[np.argmax([id in edf_name for edf_name in list_edf])])
    # Update fileID
    if name_cohort in ['isruc']:
        baseDir = [os.path.split(edf[:edf.find('subgroup')])[0]
                   for edf in list_edf]
        list_fileID = [fid[fid.find('subgroup'):-4] for fid in list_edf]
    else:
        baseDir, list_fileID = map(
            list, zip(*[os.path.split(edf[:-4]) for edf in list_edf]))

    # Depending on the cohort, subjectID is found in different ways
    if name_cohort == 'shhs':
        list_subjectID = [fileID[6:] for fileID in list_fileID]
    elif name_cohort == 'mros':
        list_subjectID = [fileID[12:] for fileID in list_fileID]
    elif name_cohort == 'wsc':
        list_subjectID = [fileID[:5] for fileID in list_fileID]
    elif name_cohort == 'ssc':
        list_subjectID = [fileID.split(sep='_')[1] for fileID in list_fileID]
    elif name_cohort == 'isruc':
        list_subjectID = ['/'.join(fid.split('/')[:2]) for fid in list_fileID]
    else:
        list_subjectID = list_fileID

    # Create empty dataframe for cohort
    df_cohort = pd.DataFrame(
        columns=['File', 'Hypnogram', 'FileID', 'SubjectID', 'Cohort', 'Partition', 'Skip', 'HypnogramLength']).fillna(0)
    df_cohort['File'] = list_edf
    df_cohort['Hypnogram'] = list_hypnogram
    df_cohort['FileID'] = list_fileID
    df_cohort['SubjectID'] = list_subjectID
    df_cohort['Cohort'] = name_cohort

    if name_cohort in ['isruc']:
        df_cohort['SubjectID'] = [subjID.replace(
            '/', '_') for subjID in df_cohort['SubjectID']]
        df_cohort['FileID'] = [fID.replace('/', '_')
                               for fID in df_cohort['FileID']]

    # Define train/eval/test split
    unique_subjects = sorted(list(set(df_cohort['SubjectID'])))
    n_subjects = len(unique_subjects)
    LOG.info('Current cohort: {: <5} | Total: {} subjects, {} EDFs'.format(
        name_cohort, n_subjects, len(list_edf)))
    seed(name_cohort[0])
    shuffle(unique_subjects)
    trainID, evalID, testID = np.split(unique_subjects, [int(
        PARTITIONS['TRAIN'] * n_subjects), int((PARTITIONS['TRAIN'] + PARTITIONS['EVAL']) * n_subjects)])
    LOG.info('{: <5} | Assigning subjects to subsets: {}/{}/{} train/eval/test'.format(
        name_cohort, len(trainID), len(evalID), len(testID)))
    for id in df_cohort['SubjectID']:
        if id in trainID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'train'
        elif id in evalID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'eval'
        elif id in testID:
            df_cohort.loc[df_cohort['SubjectID'] == id, 'Partition'] = 'test'
        else:
            print('No subset assignment for {}.'.format(id))

    # Process files
    for idx, row in df_cohort.iterrows():
        psg, hypnogram = process_file(row)
        if psg is None:
            LOG.info('{: <5} | Skipping file: {}'.format(
                name_cohort, row['FileID']))
            df_cohort.loc[idx, 'Skip'] = 1
        else:
            psg = np.concatenate([psg[mod]
                                    for mod in ['eeg', 'eog', 'emg']], axis=0)
            N = np.min(
                [len(hypnogram), psg.shape[1]])
            LOG.info('{: <5} | {} | Writing {} epochs'.format(
                name_cohort, row['FileID'], N))

            # Write H5 file for subject
            write_H5(psg, hypnogram, N, row)
            df_cohort.loc[idx, 'HypnogramLength'] = N

    return df_cohort


def main():
    LOG.info('Processing cohorts: {}'.format([*COHORTS]))
    df = []

    # Loop over the different cohorts
    for name_cohort, cohort in COHORTS.items():

        LOG.info('Processing cohort: {}'.format(name_cohort))

        if not cohort['edf'] or not os.path.exists(cohort['edf']):
            LOG.info('Skipping cohort: {}'.format(name_cohort))
            continue

        # process_cohort(current_cohort_overview, current_cohort)
        df_cohort = process_cohort(cohort, name_cohort)

        if isinstance(df_cohort, pd.DataFrame):
            filename = os.path.join(
                OUTPUT_DIRECTORY, 'csv', name_cohort + '.csv')
            df_cohort.to_csv(filename)

    LOG.info('Processing cohorts finalized.')


if __name__ == '__main__':
    main()
