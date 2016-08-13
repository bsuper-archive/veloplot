import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import utils
from numpy.linalg import lstsq

sns.set_style("whitegrid")
sns.set_palette("bright")

def nanofile_to_dataframe(filename):
    cols = ["drop", "drop", "time",
            "Fx", "Fy", "Fz",
            "Mx", "My", "Mz",
            "Ax", "Ay", "Az"
    ]
    df = pd.read_csv(filename, names=cols)
    drop_cols = [0,1,9,10,11]
    return drop_cols(df, drop_cols)

def telemetry_to_dataframe(filename):
    data_csv = utils.write_data_file_to_csv(filename)
    df = pd.read_csv(data_csv)
    drop_cols = list(range(1, 16)) # columns we drop
    return drop_columns(df, drop_cols)

def drop_columns(df, cols):
    df = df.drop(dataframe.columns[cols], axis=1)
    return df

def downsample(df, freq_high, freq_low):
    diff = freq_high/freq_low


def align_two_streams(df_high, df_low, freq_high, freq_low, flick_high, flick_low):
    nano = df_high[flick_high:]
    telem = df_low[flick_low:]
    nano = downsample(nano, freq_high, freq_low)

def least_squares_fit(M, S):
    S_squared = S * S
    S_cubed = S * S * S
    S_24 = np.zeros((S.shape[0], 8, 3))

    for i in xrange(S.shape[0]):
        for j in xrange(S.shape[1]):
            thirds = np.zeros((3,))
            thirds[0], thirds[1], thirds[2] = S[i][j], S_squared[i][j], S_cubed[i][j]
            S_24[i][j] = thirds

    S_24 = S_24.reshape((S_24.shape[0], 24))
    print "=== S_24 shape:{0} ===".format(S_24.shape)

    #C is the vector we are solving for in the S_24 * C = M equation.
    C = lstsq(S_24, M)
    print "=== least squares fit DONE ==="
    print "=== C shape: {0} ===".format(C.shape)
    return C
