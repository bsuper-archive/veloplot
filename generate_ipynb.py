import nbformat as nbf
import fnmatch
import os
from runipy.notebook_runner import NotebookRunner


def generate_notebooks(data_dir="experiment_data", overwrite=False):
    """
    Creates an iPython notebook for all data files in data_dir. If overwrite is
    false, we leave existing iPython notebook as is.
    """
    data_files = get_data_files(data_dir)
    for data_file in data_files:
        output_filename = data_file.split(".txt")[0] + ".ipynb"
        create_notebook(data_file, output_filename, overwrite=overwrite)


def get_data_files(data_dir="experiment_data"):
    """
    Finds all data files (.txt) in the data_dir.
    """
    matches = []
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*.txt'):
            matches.append(os.path.join(root, filename))
    return matches


def create_notebook(data_file, output_filename, overwrite=False):
    """
    Creates the standard tactile velociroach iPython notebook from data_file and
    writes it to data_file.
    """
    # Don't overwrite existing iPython notebooks if overwrite is false.
    if not overwrite and os.path.isfile(output_filename):
        print("Found {0} so skipping generating notebook for {1}".format(
            output_filename, data_file))
        return

    print("Generating iPython notebook {0} for {1}...".format(output_filename,
                                                              data_file))

    num_dirs_to_root = data_file.count('/')
    to_root_path = '../' * num_dirs_to_root

    nb = nbf.v4.new_notebook()

    import_cell = \
    "%matplotlib inline\n" \
    "%config InlineBackend.figure_format = 'retina'\n" \
    "import sys\n" \
    "sys.path.append('{0}')\n" \
    "import utils".format(to_root_path)

    # Because this file is run from the top directory and iPython notebooks are
    # run from the subdirectories, we must first leave the path to the root
    # empty to run the cell, then change the cell after we ran it so that we
    # can run the iPython notebooks via jupyter.
    header_cell = lambda to_root_path_ : \
    "CALIBRATION_FILE = '{0}calibration/out/cal_1_C_matrix.mat'\n" \
    "DATA_FILE = '{0}{1}'\n" \
    "utils.print_header(DATA_FILE)".format(to_root_path_, data_file)

    robot_info_cell = \
    "df = utils.process_data_files(DATA_FILE, CALIBRATION_FILE)\n" \
    "utils.plot_columns(df, [['TorqueL', 'TorqueR'], ['Left Leg Pos', 'Right Leg Pos'], ['RBEMF', 'LBEMF'], ['VMotorR', 'VMotorL'], ['PowerR', 'PowerL'], 'VBatt', 'AngleZ'], display=True, save_figure=False)"

    force_torque_cell = \
    "utils.plot_columns(df, [['Fx', 'Fy', 'Fz'], 'F_mag', ['Mx', 'My', 'Mz'], 'M_mag', ['AX', 'AY', 'AZ'], 'A_mag', ['GyroX', 'GyroY', 'GyroZ'], 'Gyro_mag'], display=True, save_figure=False)"

    nb['cells'] = [nbf.v4.new_code_cell(import_cell),
                   nbf.v4.new_code_cell(header_cell('')),
                   nbf.v4.new_code_cell(robot_info_cell),
                   nbf.v4.new_code_cell(force_torque_cell)]

    with open(output_filename, 'w') as f:
        nbf.write(nb, f)

    # Due to the terrible cross compatability between iPython version 3 and
    # version 4, we're unable to read the iPython v4 notebook we just created as
    # a v4 notebook, because of a self.nb.worksheets attribute, so we must read
    # it as a v3 notebook. Then, there's an issue with runipy for writing a v3
    # notebook directly as a v4, so we must first save as v3 and then read it
    # again and save as v4.
    notebook = nbf.read(output_filename, 3)
    r2 = NotebookRunner(notebook)
    r2.run_notebook()
    # See comment above header_cell.
    r2.nb['worksheets'][0]['cells'][1] = nbf.v3.new_code_cell(
        header_cell(to_root_path))
    nbf.write(r2.nb, output_filename, version=3)
    notebook = nbf.read(output_filename, 3)
    nbf.write(notebook, output_filename, version=4)

    print("Generated iPython notebook {0} for {1}".format(output_filename,
                                                          data_file))


if __name__ == "__main__":
    generate_notebooks(overwrite=False)
