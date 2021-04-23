import argparse
import math
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model.metrics as module_metric
from src.utils.config import process_config
from src.utils.ensure_dir import ensure_dir
from src.utils.factory import create_instance
from src.utils.logger import Logger


def main(config, resume):
    test_logger = Logger()

    # Choose subsets
    subsets = ['test']

    # build model architecture
    model = create_instance(config.network)(config)
    print(model)

    # Load checkpointed model
    print(f'Loading checkpoint: {resume} ...')
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config.trainer.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    else:  # HACK: model was trained on multiple GPUs, this removes the 'module' part of the state_dict keys.
        state_dict_old = state_dict.copy()
        state_dict = {k.replace('module.', ''): v for k, v in state_dict_old.items()}
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Prepare directory
    _, chkpoint = os.path.split(resume)
    exp_dir = config.trainer.log_dir
    prediction_dir = os.path.join(exp_dir, 'predictions-' + chkpoint.split('.')[0])
    ensure_dir(prediction_dir)

    # Loop over subsets
    subjects = {subset: None for subset in subsets}
    df_total = []
    for subset in subsets:

        # Setup data_loader instances
        dataset = create_instance(config.data_loader)(config, subset=subset)
        df_subset = dataset.df
        subjects_in_subset = {r.FileID: {'true': [], 'pred': []} for _, r in df_subset.iterrows()}
        data_loader = DataLoader(dataset,
                                 batch_size=dataset.batch_size,
                                 shuffle=False,
                                 num_workers=20,
                                 drop_last=False,
                                 pin_memory=True)

        # Get raw predictions
        bar = tqdm(data_loader, total=len(data_loader))
        bar.set_description(f'[ {subset.upper()} ]')
        predictions = []
        targets = []
        with torch.no_grad():
            for i, out in enumerate(bar):
                data = out['data']
                target = out['target'].cpu().numpy()
                file_id = out['fid']
                position = out['position']
                output = model(data.to(device)).cpu()
                for j, fid in enumerate(file_id):
                    if i == 0 and j == 0:
                        current_subject = fid
                    if current_subject == fid:
                        targets.append(target[j, :])
                        predictions.append(output[j, :, :].softmax(dim=0).numpy())
                    else:
                         # Save predictions as pickles with true and predicted labels for each subject as a separate file. Predictions are softmaxes every 1 second.
                        with open(os.path.join(prediction_dir, current_subject + '.pkl'), 'wb') as handle:
                            pickle.dump({'targets': targets, 'predictions': predictions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # Reset variables
                        current_subject = fid
                        targets = []
                        predictions = []

            # This just saves the last prediction.
        with open(os.path.join(prediction_dir, current_subject + '.pkl'), 'wb') as handle:
            pickle.dump({'targets': targets, 'predictions': predictions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # # Save predictions as pickles with true and predicted labels for each subject as a separate file. Predictions are softmaxes every 1 second.
        # exp_dir, chkpoint = os.path.split(resume)
        # prediction_dir = os.path.join(exp_dir, 'predictions-' + chkpoint.split('.')[0])
        # ensure_dir(prediction_dir)
        # bar = tqdm(subjects_in_subset.items())
        # bar.set_description(f'[ {subset.upper()} ] Saving predictions as pickles')
        # for k, v in bar:
        #     with open(os.path.join(prediction_dir, k + '.pkl'), 'wb') as handle:
        #         pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Append subset dataframe to total
        df_total.append(df_subset)

        # Free up memory
        del dataset, data_loader, subjects_in_subset

    # Free up memory
    del model

    # Concatenate subset dataframes and save total dataframe
    df_total = pd.concat(df_total)
    df_total.to_csv(os.path.join(prediction_dir, 'overview.csv'))

    # return
    print('Running predictions based off smaller windows')

    # Create a dataframe for each eval window
    df_pred = []
    evaluation_windows = [1, 3, 5, 10, 15, 30]
    confmat_subject = {fid: {eval_window: None for eval_window in evaluation_windows} for fid in df_total['FileID'].values}
    confmat_total = {eval_window: np.zeros((5, 5)) for eval_window in evaluation_windows}
    for eval_window in evaluation_windows:
        df = pd.DataFrame()
        df['FileID'] = df_total['FileID'].values
        df['Subset'] = df_total['Partition'].values
        df['Cohort'] = df_total['Cohort'].values
        df['Experiment'] = config.exp.name
        df['Window'] = f'{eval_window} s'
        for idx, row in tqdm(df.iterrows(), total=len(df)):

            # Get the true and predicted stages
            fid = row.FileID
            with open(os.path.join(prediction_dir, fid + '.pkl'), 'rb') as handle:
                labels = pickle.load(handle)
            t = np.concatenate(labels['targets'], axis=0)
            p = np.concatenate(labels['predictions'], axis=1)
            # subset = row.Subset
            # t = np.concatenate(subjects[subset][fid]['true'], axis=0)
            # p = np.concatenate(subjects[subset][fid]['pred'], axis=1)

            # Extract the metrics
            acc = metrics.accuracy_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0))
            bal_acc = metrics.balanced_accuracy_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0))
            kappa = metrics.cohen_kappa_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0))
            f1 = metrics.f1_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0), average='macro')
            prec = metrics.precision_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0), average='macro')
            recall = metrics.recall_score(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0), average='macro')
            mcc = metrics.matthews_corrcoef(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0))

            # Assign metrics to dataframe
            df.loc[idx, 'Overall accuracy'] = acc
            df.loc[idx, 'Balanced accuracy'] = bal_acc
            df.loc[idx, 'Kappa'] = kappa
            df.loc[idx, 'F1'] = f1
            df.loc[idx, 'Precision'] = prec
            df.loc[idx, 'Recall'] = recall
            df.loc[idx, 'MCC'] = mcc

            # Get stage-specific metrics
            precision, recall, f1, support = metrics.precision_recall_fscore_support(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0), labels=[0, 1, 2, 3, 4])

            # Assign to dataframe
            for stage_idx, stage in zip([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM']):
                df.loc[idx, f'F1 - {stage}'] = f1[stage_idx]
                df.loc[idx, f'Precision - {stage}'] = precision[stage_idx]
                df.loc[idx, f'Recall - {stage}'] = recall[stage_idx]
                df.loc[idx, f'Support - {stage}'] = support[stage_idx]

            # Get confusion matrix
            C = metrics.confusion_matrix(t[::eval_window], np.mean(p.reshape(5, -1, eval_window), axis=2).argmax(axis=0), labels=[0, 1, 2, 3, 4])
            confmat_subject[fid][eval_window] = C
            confmat_total[eval_window] += C

        # Update list
        df_pred.append(df)

    # Finalize dataframe
    df_pred = pd.concat(df_pred)

    # Save dataframe
    # exp_dir, chkpoint = os.path.split(resume)
    df_pred.to_csv(os.path.join(prediction_dir, 'predictions.csv'))

    # Save confusion matrices to pickle
    C = {'total': confmat_total, 'subject-specific': confmat_subject}
    with open(os.path.join(prediction_dir, 'confusionmatrix.pkl'), 'wb') as handle:
        pickle.dump(C, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='deep-sleep-pytorch')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='Path to configuration file (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # DEBUGGING:
    # args.config = 'src/configs/exp02-frac100.yaml'
    # args.resume = 'experiments/exp02-frac100/0526_223319/checkpoint-epoch19.pth'
    # args.config = 'src/configs/exp01-hu1024-sgd-clr.yaml'
    # args.resume = 'experiments/exp01-hu1024-sgd-cycliclr/0505_084225/checkpoint-epoch39.pth'
    # args.device = '0'

    if args.config:
        # load config file
        config = process_config(args.config)

        # setting path to save trained models and log files
        path = os.path.join(config.trainer.save_dir, config.exp.name)

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.yaml', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
