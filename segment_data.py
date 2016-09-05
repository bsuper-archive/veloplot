import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

import utils

parser = argparse.ArgumentParser(
    description='Segments each data into three parts: no-contact, contact,\
     no-contact. Saves them into a data folder.')
parser.add_argument(
    '--input_dir',
    default='./input/',
    help='Directory where experiments are stored')
parser.add_argument(
    '--output_dir',
    default='./data/',
    help='Directory to save all the segmented files')
parser.add_argument(
    '--file', default='none', help='Path to the file you want to segment')
args = parser.parse_args()


def segment_data_gui(df,
                     columns=[["Fx", "Fy", "Fz"], "F_mag", ["Mx", "My", "Mz"],
                              "M_mag", ["AX", "AY", "AZ"], "A_mag"],
                     output_dir="segments/",
                     output_filename="segmented"):
    """
    Columns - list of columns to plot with respect to time
    """

    figure, axarr = plt.subplots(len(columns))

    d = {'num_events_picked': 0, 'segment_begin': 0}

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        d['num_events_picked'] += 1
        if d['num_events_picked'] == 2:
            print "Segment End: ", xdata[ind][0]
            segment_df(
                df,
                d["segment_begin"],
                xdata[ind][0], output_dir=output_dir,
                output_filename=output_filename)
            plt.close()
        else:
            print "Segment Begin: ", xdata[ind][0]
            d["segment_begin"] = xdata[ind][0]

    for i in range(len(columns)):
        if type(columns[i]) == list:
            for col in columns[i]:
                axarr[i].plot(df["time"], df[col], label=col, picker=1)
                axarr[i].set_ylabel(utils.ylabels[col])
                axarr[i].set_title(utils.titles[col])
        else:
            axarr[i].plot(
                df["time"], df[columns[i]], label=columns[i], picker=1)
            axarr[i].set_ylabel(utils.ylabels[columns[i]])
            axarr[i].set_title(utils.titles[columns[i]])
        axarr[i].set_xlim([0, df["time"].max()])
        axarr[i].set_xlabel("Time (s)")
        axarr[i].legend(loc='upper right')
    figure.set_size_inches(12, int(2 * len(columns)))
    figure.canvas.mpl_connect('pick_event', onpick)
    plt.tight_layout()
    # Calling this multiple times results in TclError
    plt.show()  # http://stackoverflow.com/questions/2397791/how-can-i-show-figures-separately-in-matplotlib


def do_segmentation(data_file, calibration_file, output_dir="segments/"):
    calibration = loadmat(calibration_file)['N']
    data_csv = utils.write_data_file_to_csv(data_file)
    df = pd.read_csv(data_csv)
    df = utils.process_data(df, calibration)
    print df.columns
    experiment_name = data_file.split('/')[-1]
    segment_data_gui(
        df,
        output_dir=output_dir,
        output_filename=experiment_name.replace("txt", "").replace(".", ""))


def segment_df(df,
               segment_begin,
               segment_end,
               output_dir="segments/",
               output_filename="segmented"):
    segment_begin, segment_end = int(segment_begin * 1000), int(segment_end *
                                                                1000)
    control_seg1 = df.iloc[:segment_begin]
    activity_seg = df.iloc[segment_begin:segment_end].reset_index()
    control_seg2 = df.iloc[segment_end:].reset_index()
    output_dir = args.output_dir  #./data/
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exp_num = output_filename.replace("sliding", "")
    control_seg1.to_csv(output_dir + "ctl1-" + exp_num + ".csv")
    activity_seg.to_csv(output_dir + "act-" + exp_num + ".csv")
    control_seg2.to_csv(output_dir + "ctl2-" + exp_num + ".csv")


def segment_datafiles(datafile_folder,
                      calibration_file="calibration/out/cal_1_C_matrix.mat"):
    print glob.glob(datafile_folder + "*")
    for f in glob.glob(datafile_folder + "*"):
        do_segmentation(f, calibration_file, args.output_dir)


if __name__ == "__main__":

    if args.file != 'none':
        print "Segmenting Data File: {0}".format(args.file)
        do_segmentation(
            args.file,
            calibration_file="calibration/out/cal_1_C_matrix.mat",
            output_dir=args.output_dir)
    else:
        print "Segmenting Experiment Data Files...\n"
        input_dir = args.input_dir
        segment_datafiles(input_dir)

    print "Done."
