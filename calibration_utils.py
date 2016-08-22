import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import utils
from numpy.linalg import lstsq

sns.set_style("whitegrid")
sns.set_palette("bright")
""" Reads a nano sensor file with the given filename
    outputs a pandas dataframe object with the first
    two columns dropped since we dont need them.
"""


def nanofile_to_dataframe(filename):
    """
    Converts nanofile to dataframe.
    """
    column_names = ["drop", "drop", "time", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
                    "Ax", "Ay", "Az"]
    df = pd.read_csv(filename, names=column_names)
    cols = [0, 1, 9, 10, 11]
    return drop_columns(df, cols)


def telemetry_to_dataframe(filename):
    """
    Converts telemetry to dataframe
    """
    data_csv = utils.write_data_file_to_csv(filename)
    df = pd.read_csv(data_csv)
    drop_cols = list(range(1, 16))  # columns we drop
    return drop_columns(df, drop_cols)


def drop_columns(df, cols):
    return df.drop(df.columns[cols], axis=1)


def downsample_with_frequency(df, freq_high, freq_low):
    """
    Args:
        df: dataframe to downsample
        freq_high: high frequency that df is collected eg. 10 for 10kHz
        low_high: low frequency that other df is collected eg: 1 for 1kHz
    Returns:
        A numpy matrix downsampled to match the low frequency.
    """
    batch_size = freq_high / freq_low
    batch = df.groupby(np.arange(len(df)) // batch_size)
    out_shape = (len(batch), df.shape[1] - 1)  # omit the time column from data
    out = np.zeros(out_shape)
    for i, data in batch:
        out[i, :] = data.mean(axis=0).as_matrix(
            columns=["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
    return out


def align_two_streams(df_high, df_low, freq_high, freq_low, flick_high,
                      flick_low):
    """
    Args:
        flick_high: index of the flick in the high freq datastream
        flick_low: index of the flick in the low freq datastream
    Returns:
        Aligned versions of data files in numpy matrix format.
    """
    nano = df_high[flick_high:]
    telem = df_low[flick_low:]
    nano_np = downsample_with_frequency(nano, freq_high, freq_low)
    telem_np = telem.as_matrix(
        columns=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"])
    return nano_np, telem_np


def least_squares_fit(M, S):
    """
    Least squares fit to fit two datastreams to create calibration matrix C.
    Args:

    Returns:
        Calibration matrix C.
    """
    S_squared = S * S
    S_cubed = S * S * S
    S_24 = np.zeros((S.shape[0], 8, 3))

    for i in xrange(S.shape[0]):
        for j in xrange(S.shape[1]):
            thirds = np.zeros((3, ))
            thirds[0], thirds[1], thirds[2] = S[i][j], S_squared[i][
                j], S_cubed[i][j]
            S_24[i][j] = thirds

    S_24 = S_24.reshape((S_24.shape[0], 24))
    print "=== S_24 shape:{0} ===".format(S_24.shape)

    #C is the vector we are solving for in the S_24 * C = M equation.
    C = lstsq(S_24, M)[0]
    print "=== least squares fit DONE ==="
    print "=== C shape: {0} ===".format(C.shape)
    return C
