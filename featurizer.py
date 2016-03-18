import numpy as np
import utils
from scipy import fftpack, signal, stats
import itertools
import glob
import pandas as pd

def segment(data, window_size=512, overlap=.5, padding=None):
    """
    Segment data by WINDOW SIZE with OVERLAP
    * Padding not yet completed
    """
    windows = []
    next_window_offset = int((1 - overlap) * window_size)
    for i in range(0, data.shape[0], next_window_offset):
        if not padding and i + window_size > data.shape[0]:
            break
        windows.append(data[i:i + window_size])
    return windows

SAMPLING_FREQUENCY = 1000.
SAMPLING_INTERVAL = 1. / SAMPLING_FREQUENCY

def fft_freq(data, window_func=signal.hanning):
    w = window_func(data.size)
    sig_fft = fftpack.fft(data * w)
    freq = fftpack.fftfreq(sig_fft.size, d=SAMPLING_INTERVAL)
    freq = freq[range(data.size / 2)]
    sig_fft = sig_fft[range(data.size / 2)]
    return sig_fft, freq

def rfft_freq(data, window_func=signal.hanning):
    w = window_func(data.size)
    sig_fft = fftpack.rfft(data * w)
    freq = fftpack.rfftfreq(sig_fft.size, d=SAMPLING_INTERVAL)
    freq = freq[range(data.size / 2)]
    sig_fft = sig_fft[range(data.size / 2)]
    return sig_fft, freq

def freq_domain_entropy(sig_fft):
    psd = (sig_fft ** 2) / sig_fft.shape[0]
    psd = psd / np.sum(psd)
    psd = map(lambda p: -p * np.log(p), psd)
    return np.sum(psd)

def energy(sig_fft):
    return np.sum(np.abs(sig_fft) ** 2)

def pairwise_correlation(a, b):
    return np.correlate(a, b)

def featurize(df_seg):
    features = []

    cols = ["Fx", "Fy", "Fz", "F_mag", "Mx", "My", "Mz", "M_mag"]
    for col in cols:
        # statistical metrics
        features.append(np.max(df_seg[col]))
        features.append(np.min(df_seg[col]))
        features.append(np.mean(df_seg[col]))
        features.append(np.std(df_seg[col]))
        features.append(stats.skew(df_seg[col]))
        features.append(stats.kurtosis(df_seg[col]))

        # frequency domain
        sig_fft, freq = fft_freq(df_seg[col])
        features.append(freq_domain_entropy(sig_fft))
        features.append(energy(sig_fft))

    f_cols = ["Fx", "Fy", "Fz"]
    m_cols = ["Mx", "My", "Mz"]
    for col_set in [f_cols, m_cols]:
        for pair in itertools.combinations(col_set, 2):
            features.append(pairwise_correlation(df_seg[pair[0]], df_seg[pair[1]])[0])

    return np.array(features)

#########################################
# PERFORM FEATURIZATION
#########################################

DATA_FOLDER = "data/"
CTL_FILES = DATA_FOLDER + "ctl*.csv"
ACT_FILES = DATA_FOLDER + "act*.csv"

def create_feature_vector(data_files, label=0):
    lst = []
    labels = []
    for f in glob.glob(data_files):
        df = pd.read_csv(f)
        df_segs = segment(df)
        for df_seg in df_segs:
            lst.append(featurize(df_seg))
            labels.append(label)
    return np.array(lst), np.array(labels)

def get_feature_vector(ctl_files, act_files):
    ctl_features, ctl_labels = create_feature_vector(ctl_files, label=0)
    act_features, act_labels = create_feature_vector(act_files, label=1)
    import pdb; pdb.set_trace()
    features, labels = np.vstack((ctl_features, act_features)), np.concatenate((ctl_labels, act_labels))
    # add here to write these two as two .mat files to data/ dir to keep
    return features, labels
