import numpy as np


def rolling_window(a, window, step):
    """
    Make an ndarray with a rolling window of the last dimension with a given step size.

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window
    step : int
        Size of steps between windows

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size window.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
          [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
          [ 6.,  7.,  8.]])

   """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + ((a.shape[-1] - window) // step + 1, window)
    strides = a.strides[:-1] + (step * a.strides[-1], a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def segment_signal(signal, segment_length, segment_step):
    return rolling_window(signal, segment_length, segment_step)


def segment_and_sequence_psg(segmentation, fs, psg, hypnogram):
    psg_seg = {key: [] for key in psg.keys()}
    num_segments_in_sequence = int(
        segmentation['SEQUENCE_DURATION_MIN'] / segmentation['EPOCH_LENGTH_SEC'] * 60)
    for chn in psg_seg.keys():
        segment_length_sec = segmentation['SEGMENT_LENGTH'][chn]
        segment_overlap = segmentation['SEGMENT_OVERLAP'][chn]
        segment_length = int(segment_length_sec * fs)
        segment_step = int((segment_length_sec - segment_overlap) * fs)

        psg_seg[chn] = segment_signal(psg[chn], segment_length, segment_step)

        # # Create sequences
        # psg_seg[chn] = np.reshape(psg_seg[chn], [psg_seg[chn].shape[0], num_segments_in_sequence, -1, segment_length])

    # Segment the hypnogram
    hypno_seg = segment_signal(
        hypnogram, num_segments_in_sequence, num_segments_in_sequence)

    # # If the signals and hypnogram are of different lengths, we assume that the start time is fixed for both,
    # # so we trim the ends
    trim_length = np.min([hypno_seg.shape[0], psg_seg['eeg'].shape[1]])
    hypno_seg = hypno_seg[:num_segments_in_sequence *
                          (trim_length//num_segments_in_sequence), :]
    psg_seg = {chn: sig[:, :num_segments_in_sequence *
                        (trim_length//num_segments_in_sequence), :] for chn, sig in psg_seg.items()}

    # # Need to make sure the duration is divisible by the sequence length
    # psg_seg = {chn: np.reshape(sig, [sig.shape[0], -1, num_segments_in_sequence, segment_length]) for chn, sig in psg_seg.items()}
    # hypno_seg = np.reshape(hypno, [-1, num_segments_in_sequence])
    #
    # # Tranpose so that we have "batch size" ie. number of sequences in the first dimension, time step in the second
    # # dimension (ie. number of segments in sequence), number of channels, number of features (ie. 1), and number of time
    # # steps in segment (N, T, C, H, W)
    # psg_seg = {chn: np.transpose(sig, axes=[1, 0])}

    return psg_seg, hypno_seg


def segmentPSG(segmentation, fs, psg):
    psg_seg = {key: [] for key in psg.keys()}

    for chn in psg_seg.keys():
        segmentLength_sec = segmentation['SEGMENT_LENGTH'][chn]
        segmentOverlap = segmentation['SEGMENT_OVERLAP'][chn]
        segmentLength = int(segmentLength_sec * fs)
        segmentStep = int((segmentLength_sec - segmentOverlap) * fs)

        psg_seg[chn] = segment_signal(psg[chn], segmentLength, segmentStep)

    return psg_seg
