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

# Ignore the mplDeprecation warnings that arise when using plt.ginput.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
"""
Segments each data file into multiple segments and assign diff labels to each
segment.

Usage:
    python segment_data.py --input {filepath or directory} \
    --output_dir {directory}

Example:
    python segment_data.py \
    --input experiment_data/yellow_roach/drag_experiments \
    --output_dir terrain_identification

To segment:

1. Select each segment by clicking its beginning point followed by its end
point. Each segment must be sequential and cannot overlap. There should be
an even number of points selected, 2 for each segment. If there is an error
while segmenting, such as an odd number of points selected, you have the
option of redoing the segmenting, by entering [y]es on the error prompt.

2. After an even number of sequential points are selected, you have the
option of viewing the chosen segments on the graph by pressing [d], redoing
the segmentation procedure by pressing [r], quitting via [q], and keeping
the points via [y]. If you choose to keep the points, you will be prompted
to assign labels to each segment in order, starting from segment 0.
"""

parser = argparse.ArgumentParser(
    description='Performs multiple label segmentation. Each data file can have'
    'zero or more segments.')
parser.add_argument(
    '--input', default=None, help='Directory where telemetry files are stored')
parser.add_argument(
    '--output_dir',
    default='terrain_identification',
    help='Directory to save all the segmented files')
args = parser.parse_args()


def get_data_files(input_data, telemetry_suffix="_trial_imudata.txt"):
    """
    Returns a list of data files' file paths.
    """
    data_files = []
    if os.path.isfile(input_data) and input_data.endswith(telemetry_suffix):
        data_files = [input_data]
    elif os.path.isdir(input_data):
        data_files = []
        for root, dirnames, filenames in os.walk(input_data):
            for filename in fnmatch.filter(filenames,
                                           "*{0}".format(telemetry_suffix)):
                data_files.append(os.path.join(root, filename))
    else:
        print("input_data is not a file that ends with {0} or is a directory.")
    return data_files


def get_seg_fname(folder, experiment_name, seg_idx, label):
    """
    Gets the project relative filename. If folder='out', experiment_name='test',
    seg_idx=1, label='2', the seg_fname is out/test-seg-1-label-2.csv.
    """
    return os.path.join(folder, "{0}-seg-{1}-label-{2}.csv".format(
        experiment_name, seg_idx, label))


def segment(input_data,
            output_dir="terrain_identification",
            calibration_file="calibration/out/C.mat",
            telemetry_suffix="_trial_imudata.txt"):
    """
    Main method that performs the segmentation.

    Args:
        input_data: string that is either a telemetry filepath or a directory
            that contains telemetry files. The segmentation is performed on the
            telemetry file(s). If it is a file, it must end with
            telemetry_suffix. If it is a directory, the directory is recursively
            searched for files that end with telemetry_suffix.
        output_dir: string that is the directory relative to the project path
            where the segmented data is stored. Each segment is stored in
            output_dir/label/.
        calibration_file: calibration matrix used for calibrating the robot.
        telemetry_suffix: The suffix of telemetry files. Typically, the
            telemetry files are stored as {date}_{time}_trial_imudata.txt.
    """
    data_files = get_data_files(input_data, telemetry_suffix)

    if not data_files:
        print("No data files to segment. Exiting.")
        return

    for data_file in data_files:
        print("Segmenting {0}".format(data_file))
        experiment_name = os.path.basename(data_file).strip(".txt")
        df = utils.process_data_files(data_file, calibration_file)
        segments = segmenter_gui(df)
        segment_df_and_save_to_csv(df, segments, experiment_name, output_dir)
    print("Segmentation completed successfully.")


def segment_df_and_save_to_csv(df,
                               segments,
                               experiment_name,
                               output_dir,
                               default_label='0',
                               telemetry_sample_rate=1000):
    """
    Segments the data frame using the segments list and saves the segments to
    the appropriate directory based on their labels.
    """
    # Data assigned the default label will be stored in
    # output_dir/default_label.
    control_folder = os.path.join(output_dir, default_label)
    if not os.path.isdir(control_folder):
        os.makedirs(control_folder)

    # Mark each segment so that different segments that have the same label in
    # the current data file have different filenames when saved.
    seg_idx = 0
    # Keep track of the last segment and assign the data between the last
    # segment to the current segment the default value.
    last_segment_end_idx = 0
    for seg_begin, seg_end, seg_label in segments:
        curr_seg_begin_idx = int(seg_begin * telemetry_sample_rate)
        curr_seg_end_idx = int(seg_end * telemetry_sample_rate)

        # Data between the end of the last segment and the current segment is
        # assigned the default value.
        control_seg = df.iloc[last_segment_end_idx:curr_seg_begin_idx]
        # Data in the current segment will be assigned the label from seg_label.
        activity_seg = df.iloc[curr_seg_begin_idx:curr_seg_end_idx]

        # Save the control_seg to
        # output_dir/default_label/exp_name-seg-seg_idx-label-default_label.csv
        control_seg.to_csv(
            get_seg_fname(control_folder, experiment_name, seg_idx,
                          default_label),
            index=False)
        seg_idx += 1

        # Save the activity_seg to
        # output_dir/seg_label/exp_name-seg-seg_idx-label-seg_label.csv
        segment_folder = os.path.join(output_dir, seg_label)
        if not os.path.isdir(segment_folder):
            os.makedirs(segment_folder)
        activity_seg.to_csv(
            get_seg_fname(segment_folder, experiment_name, seg_idx, seg_label),
            index=False)
        seg_idx += 1

        # Update the last segment end index.
        last_segment_end_idx = curr_seg_end_idx

    # Handle the data after the last segment to the end of the experiment.
    control_seg = df.iloc[last_segment_end_idx:]
    control_seg.to_csv(
        get_seg_fname(control_folder, experiment_name, seg_idx, default_label),
        index=False)


def segmenter_gui(df,
                  columns=[["Fx", "Fy", "Fz"], "F_mag", ["Mx", "My", "Mz"],
                           "M_mag", ["AX", "AY", "AZ"], "A_mag"]):
    """
    Renders the segmenting interface and records the segments obtained from
    clicking on the plot.

    Returns:
        segments - a list tuples, where each tuple contains
            (segment begin (float), segment end (float), segment label (str)).
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
            if pairs:
                print("Selected segments: ")
                for i, p in enumerate(pairs):
                    print("{0}: {1}".format(i, p))
            else:
                print("No segments selected. If this is right, all points " + \
                " in the data file will be assigned the default label.")
            while True:
                msg = "Enter [y]es to keep, [d]isplay to view the selected" + \
                " segments, [r] to redo segmenting, [q] to quit: "
                command = raw_input(msg)
                # Keep the selected segments. Assign labels to each segment
                # before returning.
                if command.lower().startswith("y"):
                    segments = []
                    if pairs:
                        # plt.ion enables matplotlib interactive mode so we can
                        # have access to the terminal while showing the plot.
                        plt.ion()
                        # Show the selected segments on the plot.
                        utils.plot_columns(
                            df, [["Fx", "Fy", "Fz"], "F_mag",
                                 ["Mx", "My", "Mz"], "M_mag"],
                            display=False,
                            save_figure=False,
                            figsize=(12, 10),
                            color_intervals=pairs)
                        # Have the user assign a label to each segment.
                        for i, p in enumerate(pairs):
                            label = raw_input(
                                "Assign an integer label to the " + \
                                "segment {0} - {1}: ".format(i, p))
                            segments.append((p[0], p[1], label))
                        # Close the plot after the user assigns labels to all
                        # segments.
                        plt.close()
                        # Disable matplotlib interactive mode.
                        plt.ioff()
                    else:
                        print("No segments selected. The default label " + \
                        "will be assigned to all data in this data file.")
                    return segments
                # Display the segments.
                elif command.lower().startswith("d"):
                    print("Displaying the selected segments. Click the red " + \
                        " button to close the plot.")
                    utils.plot_columns(
                        df,
                        columns,
                        display=True,
                        save_figure=False,
                        figsize=(12, 10),
                        color_intervals=pairs)
                # Redo segmentation.
                elif command.lower().startswith("r"):
                    break
                # Quit.
                elif command.lower().startswith("q"):
                    return []


if __name__ == "__main__":
    if args.input:
        segment(args.input, output_dir=args.output_dir)
    else:
        print("Please specify a file or directory to begin segmentation.")
