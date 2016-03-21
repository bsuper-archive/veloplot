# veloplot
Methods for Processing and Plotting Tactile Shell Velociroach Data

###Install Dependencies
Install virtualenv, if necessary. Virtual Environments allow you to create a sandbox where you can install only project-specific dependencies.

`pip install virtualenv`

Create a virtualenv, commonly called `venv`.

`virtualenv venv`

Activate the virtualenv created.

`source venv/bin/activate`

Thats it! Now you are ready to install project dependencies.

Install dependencies

`pip install -r requirements.txt`

###Install Tensorflow
You have probably seen in the previous step that tensorflow might not have been installed properly using requirements.txt. If that is the case you will have to install it on your own. Here is a quick way to do it:

#### MAC OSX (CPU Only)
Activate venv if you haven't already

`source venv/bin/activate`

Install latest Tensorflow using pip

`pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl`

Go to this [site](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#test-the-tensorflow-installation) to test your installation

### Use iPython Notebook
`jupyter notebook`

### Files

#### [utils.py](https://github.com/bsuper/veloplot/blob/master/utils.py)
`utils.process_data_files(data_file, calibration_file)` takes in a telemetry data_file (.txt) and a calibration file (e.g. N_matrix_trial9.mat) and returns a Pandas DataFrame, a Python dictionary-like object, with variables:

`['time', 'Right Leg Pos', 'Left Leg Pos', 'Commanded Right Leg Pos',
       'Commanded Left Leg Pos', 'DCR', 'DCL', 'GyroX', 'GyroY', 'GyroZ',
       'AX', 'AY', 'AZ', 'RBEMF', 'LBEMF', 'VBatt', 'S1', 'S2', 'S3',
       'S4', 'S5', 'S6', 'S7', 'S8', 'PowerR', 'PowerL', 'Energy',
       'TorqueR', 'TorqueL', 'VMotorR', 'VMotorL', 'AngleZ', 'Fx', 'Fy',
       'Fz', 'Mx', 'My', 'Mz', 'F_mag', 'A_mag', 'M_mag', 'Gyro_mag']`
       
These variables are accessible in the following fashion.

```python
import utils
DATA_FILE = "input/sliding9.txt"
CALIBRATION_FILE = "N_matrix_trial9.mat"
df = utils.process_data_files(DATA_FILE, CALIBRATION_FILE)
df['Fx']
```

Print header

```python
import utils
DATA_FILE = "input/sliding9.txt"
utils.print_header(DATA_FILE)
```

Plot columns, display it, and save to out/basic.png

```python
import utils
DATA_FILE = "input/sliding9.txt"
CALIBRATION_FILE = "N_matrix_trial9.mat"
df = utils.process_data_files(DATA_FILE, CALIBRATION_FILE)
utils.plot_columns(df, [["TorqueL", "TorqueR"], ["Left Leg Pos", "Right Leg Pos"], ["RBEMF", "LBEMF"], ["VMotorR", "VMotorL"], ["PowerR", "PowerL"], "VBatt", "AngleZ"], display=True, save_figure=True, output_dir="out/", output_filename="basic.png")
```

See this [iPython Notebook](https://github.com/bsuper/veloplot/blob/master/example_plot.ipynb) for example usage.

####[featurizer.py](https://github.com/bsuper/veloplot/blob/master/featurizer.py)

Featurizes the data. Currently, we implement a 512 sample window (~.512s) with 50% overlap. Given that the leg frequency for our sliding experiments is 4Hz, our window would capture 1-2 complete leg cycles of data. For each window, we compute the following statistical metrics: [max, min, mean, std, skew, kurtosis] and frequency domain metrics: [entropy, energy] and also pairwise correlation.

You can view an iPython notebook showing our frequency domain features of an example telemetry data [here](https://github.com/bsuper/veloplot/blob/master/disp_freq_domain_features.ipynb).

####[segment_data.py](https://github.com/bsuper/veloplot/blob/master/segment_data.py)

Segments the data into three parts: no-contact, contact, no-contact. Currently, it only separates one file at a time. We'll implement segmenting multiple files at a time ASAP. Usage will be written soon.

####[classify.py](https://github.com/bsuper/veloplot/blob/master/classify.py)

Classifies the data. Currently, four models are implemented: Random Forests, Gradient Boosted Trees, RBF SVM, and a 2 hidden layer Neural Network with 100 nodes in each hidden layer. All of their average 10-fold cross validation accuracy are ~93-94%. Random forests and gradient boosted trees also give feature importances.

You can see the results [here](https://github.com/bsuper/veloplot/blob/master/classification_results.ipynb).
