import nbformat as nbf
import fnmatch
import os
from runipy.notebook_runner import NotebookRunner
from IPython.nbformat.current import read, write

def generate_notebooks(data_dir="experiment_data"):
    data_files = get_data_files(data_dir)
    for data_file in data_files:
        output_filename = data_file.split(".txt")[0] + ".ipynb"
        create_notebook(data_file, output_filename)

def get_data_files(data_dir="experiment_data"):
    matches = []
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*.txt'):
            print root, filename
            matches.append(os.path.join(root, filename))
    return matches

def create_notebook(data_file="../experiment_data/new_grass/one_side_1.txt", output_filename='test.ipynb'):
    print("Generating iPython notebook {0} for {1}...".format(output_filename, data_file))

    num_dirs_to_root = data_file.count('/')
    to_root_path = '../' * num_dirs_to_root

    nb = nbf.v4.new_notebook()
    import_cell = \
    "%matplotlib inline\n" \
    "%config InlineBackend.figure_format = 'retina'\n" \
    "import sys\n" \
    "sys.path.append('{0}')\n" \
    "import utils".format(to_root_path)

    header_cell = \
    "CALIBRATION_FILE = 'input/N_matrix_trial9.mat'\n" \
    "DATA_FILE = '{0}'\n" \
    "utils.print_header(DATA_FILE)".format(data_file)

    robot_info_cell = \
    "df = utils.process_data_files(DATA_FILE, CALIBRATION_FILE)\n" \
    "utils.plot_columns(df, [['TorqueL', 'TorqueR'], ['Left Leg Pos', 'Right Leg Pos'], ['RBEMF', 'LBEMF'], ['VMotorR', 'VMotorL'], ['PowerR', 'PowerL'], 'VBatt', 'AngleZ'], display=True, save_figure=False)"

    force_torque_cell = \
    "utils.plot_columns(df, [['Fx', 'Fy', 'Fz'], 'F_mag', ['Mx', 'My', 'Mz'], 'M_mag', ['AX', 'AY', 'AZ'], 'A_mag', ['GyroX', 'GyroY', 'GyroZ'], 'Gyro_mag'], display=True, save_figure=False)"

    nb['cells'] = [nbf.v4.new_code_cell(import_cell),
                   nbf.v4.new_code_cell(header_cell),
                   nbf.v4.new_code_cell(robot_info_cell),
                   nbf.v4.new_code_cell(force_torque_cell)]

    with open(output_filename, 'w') as f:
        nbf.write(nb, f)

    notebook = read(open(output_filename), 'json')
    r = NotebookRunner(notebook)
    r.run_notebook()
    write(r.nb, open(output_filename, 'w'), 'json')

    print("Generated iPython notebook {0} for {1}".format(output_filename, data_file))

if __name__ == "__main__":
    generate_notebooks()
