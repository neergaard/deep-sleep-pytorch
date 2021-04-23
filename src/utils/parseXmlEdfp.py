"""
    @authors: stanislas chambon / Alexandre Gramfort
    goal: edf annotation reader
    Modified: Alexander Neergaard Olesen, Stanford University, 2018
"""

import re

import numpy as np
import pandas as pd
import xmltodict


def read_edf_annotations(fname, annotation_format="edf/edf+"):
    """read_edf_annotations

    Parameters:
    -----------
    fname : str
        Path to file.

    Returns:
    --------
    annot : DataFrame
        The annotations
    """

    with open(fname, 'r', encoding='utf-8',
              errors='ignore') as annotions_file:
        tal_str = annotions_file.read()

    if "edf" in annotation_format:
        if annotation_format == "edf/edf+":
            exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
                  '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
                  '(\x14(?P<description>[^\x00]*))?' + '(?:\x14\x00)'

        elif annotation_format == "edf++":
            exp = '(?P<onset>[+\-]\d+.\d+)' + \
                  '(?:(?:\x15(?P<duration>\d+.\d+)))' + \
                  '(?:\x14\x00|\x14(?P<description>.*?)\x14\x00)'

        annot = [m.groupdict() for m in re.finditer(exp, tal_str)]
        good_annot = pd.DataFrame(annot)
        good_annot = good_annot.query('description != ""').copy()
        good_annot.loc[:, 'duration'] = good_annot['duration'].astype(float)
        good_annot.loc[:, 'onset'] = good_annot['onset'].astype(float)

    elif annotation_format == "xml":
        annot = xmltodict.parse(tal_str)
        annot = annot['PSGAnnotation']["ScoredEvents"]["ScoredEvent"]
        good_annot = pd.DataFrame(annot)

    return good_annot


def resample_30s(annot):
    """resample_30s: to resample annot dataframe when durations are multiple
    of 30s

    Parameters:
    -----------
    annot : pandas dataframe
        the dataframe of annotations

    Returns:
    --------
    annot : pandas dataframe
        the resampled dataframe of annotations
    """

    annot["start"] = annot.Start.values.astype(np.float).astype(np.int)
    df_end = annot.iloc[[-1]].copy()
    df_end['start'] += df_end['Duration'].values.astype(np.float)
    df_end.index += 1
    annot = annot.append(df_end)
    annot = annot.set_index('start')
    annot.index = pd.to_timedelta(annot.index, unit='s')
    annot = annot.resample('30s').ffill()
    annot = annot.reset_index()
    annot['duration'] = 30.
    onset = np.zeros(annot.shape[0])
    onset[1:] = annot["duration"].values[1:].cumsum()
    annot["onset"] = onset
    return annot.iloc[:-1]  # Return without the last row (which we inserted in order to fill the last row correctly).


def parse_hypnogram(annot_f, annotation_format="edf++"):
    """parse_hypnogram: keep only annotations related to sleep stages

    Parameters:
    -----------
    annot_f : string
        The name of the annotation file
    annotation_format : string, optional (default="edf++")
        The annotation format

    Returns:
    --------
    good_annot : pandas dataframe
        dataframe of annotations related to sleep stages
    """

    annot = read_edf_annotations(annot_f, annotation_format=annotation_format)

    if annotation_format == "edf++":

        # process annot for sleep stages
        annot = annot[annot.description.str.startswith('Sleep')].reset_index()
        annot["idx_stage"] = np.arange(annot.shape[0])

        stages = pd.DataFrame()

        description = ['Sleep stage ?', 'Sleep stage W',
                       'Sleep stage 1', 'Sleep stage 2',
                       'Sleep stage 3', 'Sleep stage R']
        label = [-1, 0, 1, 2, 3, 4]
        stages["label"] = label
        stages["description"] = description
    elif annotation_format == "xml":

        # process annot for sleep stages
        annot = annot[annot["EventType"] == "Stages|Stages"]

        annot = resample_30s(annot)
        annot["idx_stage"] = np.arange(annot.shape[0])

        stages = pd.DataFrame()

        description = ['Wake|0', 'Stage 1 sleep|1',
                       'Stage 2 sleep|2', 'Stage 3 sleep|3',
                       'Stage 4 sleep|4', 'REM sleep|5']

        label = [0, 1, 2, 3, 3, 4]
        stages["label"] = label
        stages["description"] = description

    good_annot = pd.merge(
        annot, stages,
        left_on="EventConcept", right_on="description")
    good_annot = good_annot.sort_values(
        by="idx_stage").reset_index(drop=True)

    # filter non labeled epochs
    good_annot = good_annot[good_annot.label != -1]
    good_annot = good_annot.reset_index(drop=True)

    return good_annot
