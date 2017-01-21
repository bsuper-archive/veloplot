import argparse
import fnmatch
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

import utils

parser = argparse.ArgumentParser(
    description='Performs leg cycle time extraction on telemetry files.')
parser.add_argument(
    '--input', default=None, help='Telemetry file')
args = parser.parse_args()

def extract_cycles(input_data, calibration_file="calibration/out/C.mat"):
    df = utils.process_data_files(input_data, calibration_file)
    cycles = select_cycles_gui(df)
    return cycles


def select_cycles_gui(df, columns=[['TorqueL', 'TorqueR'],
                                    ['Left Leg Pos', 'Right Leg Pos'],
                                    ['RBEMF', 'LBEMF']]):

    """
    Returns:
        cycles = list of tuples where each tuple contains cycle begin time and end time
    """
    while True:
        # Render the segmenting interface and allow the users to choose points.
        error_msg = ""
        utils.plot_columns(
            df, columns, display=False, save_figure=False, figsize=(12, 10))
        try:
            print("Select segments of interest by selecting beginning and "
                  "then end point for each segment. Press [ENTER] to finish "
                  "segmenting.")
            # plt.ginput(-1) renders the plot and accepts clicks on the plot.
            # -1 means that the user can click many times on the plot and the
            # plot will close after [ENTER] is pressed.
            coords = plt.ginput(-1)
            plt.close()

            # There should be an even number of coordinates so that we can have
            # a begin and an end point.
            if len(coords) % 2:
                error_msg = (
                    "ERROR: There should be an even number "
                    "points. Enter [y]es to repeat segmenting, and any "
                    "other key to quit: ")
            elif sorted(coords) != coords:
                error_msg = ("ERROR: The x coordinates aren't in order. "
                             "Enter [y]es to repeat segmenting, and any other "
                             "key to quit: ")
        except:
            error_msg = ("ERROR: An error occured while segmenting. "
                         "Enter [y]es to repeat segmenting, and any other "
                         "key to quit: ")

        # After segmenting, handle any errors or give the user the option to
        # view the selected segments, keep the segments, or redo segmentation.
        if error_msg:
            command = raw_input(error_msg)
            if command.lower().startswith("y"):
                # Restarts the segmentation.
                continue
            else:
                return []
        else:
            # Convert [(x1,y1),(x2,y2),(x3,y3),(x4,y4),...] to
            # [(x1, x2), (x3, x4), ...]
            pairs = [(coords[i][0], coords[i + 1][0])
                     for i in range(0, len(coords), 2)]
            cycles = []
            if pairs:
                print("Selected segments: ")
                for i, p in enumerate(pairs):
                    print("{0}: {1}".format(i, p))
                    cycles.append((p[0], p[1]))
            else:
                print("No segments selected.")

            return cycles

if __name__ == "__main__":
    if args.input:
        extract_cycles(args.input)
    else:
        print("Please specify a file or directory to begin segmentation.")
