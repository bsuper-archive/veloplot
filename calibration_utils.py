import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import utils

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
