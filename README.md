# veloplot
Methods for Processing and Plotting Tactile Shell Velociroach Data

### Download dependencies
`pip install -r requirements.txt`

### Use iPython Notebook
`jupyter notebook`

### Run manually
```python
CALIBRATION_FILE = "N_matrix_trial9.mat"
DATA_FILE = "2016.03.06_19.16.20_trial_3_imudata.txt"
df = process_data_files(DATA_FILE, CALIBRATION_FILE)
plot_forces_moments(df)
plot_columns(df, [["TorqueL", "TorqueR"], ["Left Leg Pos", "Right Leg Pos"], ["Fx", "Fy", "Fz"], ["Mx", "My", "Mz"]])
```
