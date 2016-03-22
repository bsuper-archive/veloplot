import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import os
import glob
import argparse

parser = argparse.ArgumentParser(
    description='Segments each data into three parts: no-contac, contact,\
     no-contact. Saves them into a data folder.')
parser.add_argument('--input_dir', default='./input/',
    help='Directory where experiments are stored')
parser.add_argument('--output_dir', default='./data/',
    help='Directory to save all the segmented files')
args = parser.parse_args()

sns.set_style("whitegrid")
sns.set_palette("bright")

titles = {
    "AX": "Accelerations",
    "AY": "Accelerations",
    "AZ": "Accelerations",
    "A_mag": "Magnitude of Accelerations",
    "Fx": "Forces",
    "Fy": "Forces",
    "Fz": "Forces",
    "F_mag": "Magnitude of Forces",
    "Mx": "Moments",
    "My": "Moments",
    "Mz": "Moments",
    "M_mag": "Magnitude of Moments",
    "TorqueL": "Torques",
    "TorqueR": "Torques",
    "Right Leg Pos": "Leg Positions",
    "Left Leg Pos": "Leg Positions",
    "RBEMF": "Back EMF",
    "LBEMF": "Back EMF",
    "VMotorR": "VMotor",
    "VMotorL": "VMotor",
    "PowerR": "Power",
    "PowerL": "Power",
    "GyroX": "Gyro",
    "GyroY": "Gyro",
    "GyroZ": "Gyro",
    "Gyro_mag": "Speed of rotation",
    "Energy": "Energy",
    "VBatt": "VBatt",
    "AngleZ": "Anglez"
}

ylabels = {
    "AX": "Acceleration m/s^2",
    "AY": "Acceleration m/s^2",
    "AZ": "Acceleration m/s^2",
    "A_mag": "Acceleration m/s^2",
    "Fx": "Force (N)",
    "Fy": "Force (N)",
    "Fz": "Force (N)",
    "F_mag": "Force (N)",
    "Mx": "Moment (mN * m)",
    "My": "Moment (mN * m)",
    "Mz": "Moment (mN * m)",
    "M_mag": "Moment (mN * m)",
    "TorqueL": r'$\tau$ (mN * m)',
    "TorqueR": r'$\tau$ (mN * m)',
    "Right Leg Pos": "leg position (rad)",
    "Left Leg Pos": "leg position (rad)",
    "RBEMF": r'$\frac{kg * m^2}{A * s^2}$',
    "LBEMF": r'$\frac{kg * m^2}{A * s^2}$',
    "VMotorR": "VMotor",
    "VMotorL": "VMotor",
    "PowerR": "Power",
    "PowerL": "Power",
    "GyroX": "degrees/s",
    "GyroY": "degrees/s",
    "GyroZ": "degrees/s",
    "Gyro_mag": "degrees/s",
    "Energy": "Energy",
    "VBatt": "VBatt",
    "AngleZ": "Anglez"
}

def segment_data_gui(df, columns=[["Fx", "Fy", "Fz"], "F_mag", ["Mx", "My", "Mz"], "M_mag", ["AX", "AY", "AZ"], "A_mag"], output_dir="segments/", output_filename="segmented"):
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
            segment_df(df, d["segment_begin"], xdata[ind][0], output_dir=output_dir, output_filename=output_filename)
            plt.close()
        else:
            print "Segment Begin: ", xdata[ind][0]
            d["segment_begin"] = xdata[ind][0]

    for i in range(len(columns)):
        if type(columns[i]) == list:
            for col in columns[i]:
                axarr[i].plot(df["time"], df[col], label=col, picker=1)
                axarr[i].set_ylabel(ylabels[col])
                axarr[i].set_title(titles[col])
        else:
            axarr[i].plot(df["time"], df[columns[i]], label=columns[i], picker=1)
            axarr[i].set_ylabel(ylabels[columns[i]])
            axarr[i].set_title(titles[columns[i]])
        axarr[i].set_xlim([0, df["time"].max()])
        axarr[i].set_xlabel("Time (s)")
        axarr[i].legend(loc='upper right')
    figure.set_size_inches(12, int(2 * len(columns)))
    figure.canvas.mpl_connect('pick_event', onpick)
    plt.tight_layout()
    # Calling this multiple times results in TclError
    plt.show() # http://stackoverflow.com/questions/2397791/how-can-i-show-figures-separately-in-matplotlib

def do_segmentation(data_file, calibration_file, output_dir="segments/"):
    calibration = loadmat(calibration_file)['N']
    data_csv = utils.write_data_file_to_csv(data_file)
    df = pd.read_csv(data_csv)
    df = utils.process_data(df, calibration)
    print df.columns
    segment_data_gui(df, output_dir=output_dir, output_filename=data_file.replace("txt", "").replace("input", "").replace("/", ""))

def segment_df(df, segment_begin, segment_end, output_dir="segments/", output_filename="segmented"):
    segment_begin, segment_end = int(segment_begin * 1000), int(segment_end * 1000)
    control_seg1 = df.iloc[:segment_begin]
    activity_seg = df.iloc[segment_begin:segment_end].reset_index()
    control_seg2 = df.iloc[segment_end:].reset_index()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    control_seg1.to_csv(output_dir + output_filename + "_ctl1.csv")
    activity_seg.to_csv(output_dir + output_filename + "_act.csv")
    control_seg2.to_csv(output_dir + output_filename + "_ctl2.csv")

def segment_datafiles(datafile_folder, calibration_file="N_matrix_trial9.mat"):
    print glob.glob(datafile_folder + "/*")
    for f in glob.glob(datafile_folder + "/*"):
        do_segmentation(f, calibration_file)

if __name__ == "__main__":
    CALIBRATION_FILE = "N_matrix_trial9.mat"
    DATA_FILE = "input/sliding10.txt"
    do_segmentation(DATA_FILE, CALIBRATION_FILE, output_dir="sliding10/")
    # TODO: Implement passing in folders containing many data files
    # TODO: Implement running from command line
    # segment_datafiles("input/")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input', nargs='+', help='bar help')
