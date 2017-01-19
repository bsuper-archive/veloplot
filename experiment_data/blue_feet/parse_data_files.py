import glob
import sys
sys.path.append('../../')
import utils

CALIBRATION_FILE = '../../../calibration/out/C.mat'
DATA_FILE_ROOT = '../../../experiment_data/blue_feet/'

def parse_data_files(data_file_dir):
    '''
    @return a list of data frame objects given the directory
    '''
    data_files = glob.glob(data_file_dir + '*.txt')
    df_list = []
    for data_file in data_files:
        df = utils.process_data_files(data_file, CALIBRATION_FILE)
        df_list.append(df)
    return df_list

def getCOTValues(dfs, velos, times):
    '''
    @param dfs is a list of lists of data frames containing 4,6,8,10 cm experimets
    @param velos is a list of lists of velocities for each of those experiments
    @param times is a list of tuples [(start_time, end_time), ]
    @return list of lists Cost of Transport Values Per Experiment
    @return list of lists of Velocities Per Experiemnt
    '''


def run(widths, velos, times):
    dfs = []
    for width in widths:
        dfs.append(parse_data_files(DATA_FILE_ROOT + str(width) + "/"))

    getCOTValues(dfs, velos, times)

if __name__ == "__main__":
    velos = []
    times = []
    widths = [4,6,8,10]
    run(widths, velos, times)
