import mne
import re
import matplotlib.pyplot as plt
import numpy as np

# raw = mne.io.read_raw_edf("./328_WangYong/20230717{33BE4FC8-E007-4210-A882-758BC40637DC} Data 20230718143817.edf")
raw = mne.io.read_raw_edf("./ProfushionData/20221213{F2EDDB89-386C-41E2-9F29-161B888F5BD3} Data 20230802144810.edf")
ch_types_map = {}
for i in raw.ch_names:
	if i.isdigit():
		ch_types_map[i] = "seeg"
	if "ECG" in i:
		ch_types_map[i] = "ecg"
    if "EMG" in i:
		ch_types_map[i] = "emg"
    if re.match("[LR][0-9]", i):
		ch_types_map[i] = "dbs"
    if re.match("A[0-9]", i):
		ch_types_map[i] = "syst"
    if i.startswith('SPH'):
		ch_types_map[i] = "exci"
ch_renames = {'FP1':'Fp1', 'OZ':'Oz', 'POZ':'POz', 'PZ':'Pz', 'FPZ':'Fpz', 'FP2':'Fp2', 'FZ':'Fz', 'CZ':'Cz'}
raw.set_channel_types(ch_types_map)
raw.rename_channels(ch_renames)
processed = raw.resample(250.0)
processed = processed.filter(l_freq=0.5, h_freq=120.0, verbose=False).notch_filter(freqs=[50.0, 100.0], verbose=False)
# processed.load_bad_channels(bad_file="./328_WangYong/bad_channels.txt")