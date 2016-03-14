import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("bright")

def print_header(data_file):
    """
    Print the experiment information
    """
    with open(data_file, "r") as f:
        print "".join([line for line in f][:7])

def write_data_file_to_csv(data_file, output_filename=None, output_dir="tmp/"):
    """
    Create a csv file from the data file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not output_filename:
        output_filename = data_file.replace("txt", "csv").replace("/", "-")
    with open(data_file, "r") as data_f, open(output_dir + output_filename, "w") as out_f:
        data = [line for line in data_f][8:]
        # convert column line to proper csv line
        data[0] = re.sub(r"\s*\|\s*", ",", data[0])
        data[0] = re.sub(r"%\s*", "", data[0])
        out_f.write("".join(data))
    return output_dir + output_filename

def calc_force_moment(df, calibration):
    """
    Calculates force moment

    df - Pandas DataFrame from calling pd.read_csv(data_file)
    calibration - calibration matrix [shape - (24, 6)]

    S is 24 x (# num data)
    """
    S = []
    sensor_columns = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    for col in sensor_columns:
        S.extend([df[col], df[col] ** 2, df[col] ** 3])
    S = np.swapaxes(S, 0, 1)
    force_moment = np.dot(S, calibration)
    return force_moment

def delete_duplicates(force_moment):
    pass

def calibrate_to_first_k_samples(data, k=50):
    offset = np.mean(data[:k,:], axis=0)
    return data - offset

def butterworth_filter(data):
    Wn = 20. / 1000
    filter_order = 4
    num, den = signal.butter(filter_order, Wn)
    return signal.lfilter(num, den, data, axis=0)

#########################################
# PLOTTING
#########################################

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

def plot_columns(df, columns, output_dir="out/", output_filename="plots.png", display=False, save_figure=True):
    """
    Columns - list of columns to plot with respect to time
    """

    figure, axarr = plt.subplots(len(columns))

    for i in range(len(columns)):
        if type(columns[i]) == list:
            for col in columns[i]:
                axarr[i].plot(df["time"], df[col], label=col)
                axarr[i].set_ylabel(ylabels[col])
                axarr[i].set_title(titles[col])
        else:
            axarr[i].plot(df["time"], df[columns[i]], label=columns[i])
            axarr[i].set_ylabel(ylabels[columns[i]])
            axarr[i].set_title(titles[columns[i]])
        axarr[i].set_xlim([0, df["time"].max()])
        axarr[i].set_xlabel("Time (s)")
        axarr[i].legend(loc='upper right')
    figure.set_size_inches(12, int(2 * len(columns)))
    plt.tight_layout()
    if display:
        plt.show()

    if save_figure:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print "Saving image as", output_dir + output_filename
        figure.savefig(output_dir + output_filename, dpi=350)
        print "Image saved."

#########################################
# PROCESS ROBOT TELEMETRY DATA
#########################################

leg_scale = 95.8738e-6 # 16 bit to radian
vref = 3.3 # for voltage conversion
vdivide = 3.7 / 2.7 # for battery scaling
vgain = 15.0 / 47.0 # gain of differential amplifier
RMotor = 3.3 # resistance for SS7-3.3 ** need to check **
Kt = 1.41 # motor torque constant mN-m/A  ** SS7-3.3 **

# acelerometer scale in mpu6000.c set to +- 8g
# +- 32768 data
xl_scale = (1 / 4096.0) * 9.81

# gyro in mpu6000.c scale set to +-2000 degrees per second
# +- 32768
gyro_scale = (1 / 16.384) * (np.pi / 180.0)

def process_data(df, calibration, calibrate=True, k=50, leg_pos_in_radians=True):
    # convert timestamps to seconds
    df["time"] = df["time"] / 1000000.0

    # Leg Position R,L
    df["Right Leg Pos"] = df["Right Leg Pos"] * leg_scale
    df["Left Leg Pos"] = df["Left Leg Pos"] * leg_scale
    if leg_pos_in_radians:
        df["Right Leg Pos"] %= 2 * np.pi
        df["Left Leg Pos"] %= 2 * np.pi

    # Commanded Leg Position R,L
    df["Commanded Right Leg Pos"] = df["Commanded Right Leg Pos"] * leg_scale
    df["Commanded Left Leg Pos"] = df["Commanded Left Leg Pos"] * leg_scale

    # Gyro X,Y,Z
    df["GyroX"] = df["GyroX"] * gyro_scale
    df["GyroY"] = df["GyroY"] * gyro_scale
    df["GyroZ"] = df["GyroZ"] * gyro_scale

    # Accelerometer X,Y,Z
    df["AX"] = df["AX"] * xl_scale;
    df["AY"] = df["AX"] * xl_scale;
    df["AZ"] = df["AX"] * xl_scale;
    # Butterworth filter
    df["AX"] = butterworth_filter(df["AX"])
    df["AY"] = butterworth_filter(df["AY"])
    df["AZ"] = butterworth_filter(df["AZ"])

    # BackEMF R,L
    # A/D data is 10 bits, Vref+ = AVdd = 3.3V, VRef- = AVss = 0.0V
    # BEMF volts = (15K)/(47K) * Vm + vref/2 - pidObjs[i].inputOffset
    # RBEMF = -data[:,13]*vdivide*vref/1023.0
    # LBEMF = -data[:,14]*vdivide*vref/1023.0
    df["RBEMF"] = df["RBEMF"] * vref / 1024.0 / vgain  # scale A/D to 0 to 3.3 V range and undo diff amp gain
    df["LBEMF"] = df["LBEMF"] * vref / 1024.0 / vgain

    # Battery Voltage in volts
    df["VBatt"] = df["VBatt"] * vdivide * vref / 1023.0

    # Power calculation
    # i_m = (VBatt - BEMF)/R
    # V_m is just VBatt
    df["PowerR"] = np.abs((df["DCR"] / 4096.0) * df["VBatt"] * (df["VBatt"] - df["RBEMF"]) / RMotor) # P = V_m i_m x duty cycle
    df["PowerL"] = np.abs((df["DCL"] / 4096.0) * df["VBatt"] * (df["VBatt"] - df["LBEMF"]) / RMotor) # P = V_m i_m x duty cycle

    # energy calculation
    energy = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        energy[i] = energy[i - 1] + (df["PowerR"][i] + df["PowerL"][i]) * (df["time"][i] - df["time"][i - 1])
    df["Energy"] = energy

    # torque calculation
    df["TorqueR"] = (df["DCR"] / 4096.0) * Kt * (df["VBatt"] - df["RBEMF"]) / RMotor # \Tau = Kt i_m x duty cycle
    df["TorqueL"] = (df["DCR"] / 4096.0) * Kt * (df["VBatt"] - df["LBEMF"]) / RMotor # \Tau = Kt i_m x duty cycle

    # Motor Voltage, R, L
    df["VMotorR"] = df["VBatt"] * df["DCR"]
    df["VMotorL"] = df["VBatt"] * df["DCL"]

    # approximate integral of angle
    angleZ = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        angleZ[i] = angleZ[i - 1] + df["GyroZ"][i] / 1000.
    df["AngleZ"] = angleZ

    # force and moment
    force_moment = calc_force_moment(df, calibration)
    if calibrate:
        force_moment = calibrate_to_first_k_samples(force_moment, k=k)
    force_moment = butterworth_filter(force_moment)
    M_cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    for i in range(6):
        df[M_cols[i]] = force_moment[:, i]

    # force magnitude
    df["F_mag"] = np.linalg.norm(np.array([df["Fx"], df["Fy"], df["Fz"]]), axis=0)
    # acceleration magnitude
    df["A_mag"] = np.linalg.norm(np.array([df["AX"], df["AY"], df["AZ"]]), axis=0)
    # moment magnitude
    df["M_mag"] = np.linalg.norm(np.array([df["Mx"], df["My"], df["Mz"]]), axis=0)
    # speed of rotation
    df["Gyro_mag"] = np.linalg.norm(np.array([df["GyroX"], df["GyroY"], df["GyroZ"]]), axis=0)

    return df

#####################################
# Run workflow
#####################################

def process_data_files(data_file, calibration_file):
    calibration = loadmat(calibration_file)['N']
    data_csv = write_data_file_to_csv(data_file)
    df = pd.read_csv(data_csv)
    return process_data(df, calibration)

#####################################
# TESTING Code
#####################################

if __name__ == "__main__":
    CALIBRATION_FILE = "N_matrix_trial9.mat"
    DATA_FILE = "crashing_into_wall/5/2016.03.06_19.18.30_trial_5_imudata.txt"
    df = process_data_files(DATA_FILE, CALIBRATION_FILE)
    plot_columns(df, [["TorqueL", "TorqueR"], ["Left Leg Pos", "Right Leg Pos"], ["RBEMF", "LBEMF"], ["VMotorR", "VMotorL"], ["PowerR", "PowerL"], "VBatt", "AngleZ"], output_filename="basic.png")
    plot_columns(df, [["Fx", "Fy", "Fz"], "F_mag", ["Mx", "My", "Mz"], "M_mag", ["AX", "AY", "AZ"], "A_mag", ["GyroX", "GyroY", "GyroZ"], "Gyro_mag"], output_filename="FMAG.png")
