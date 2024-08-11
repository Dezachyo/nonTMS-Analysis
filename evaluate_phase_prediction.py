#%% imports
import pandas as pd
import PyQt5
import math
import numpy as np
import pathlib
import mne
from mne.stats import permutation_cluster_test,combine_adjacency,spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
import argparse

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind
import mne.stats
import scipy.stats as stats
from tqdm import tqdm
from subfunctions import get_sub_str, cprint
from subfunctions import create_phase_epochs_morlet,plot_peak_variation,plot_prestim_phase_dist_to_ax
from subfunctions import add_prestim_col, add_peak_col
from subfunctions import create_cmw, plot_wavelet

# import custom wavelet phase prediction functions
from subfunctions import compare_custom_wavelet_prediction, compute_phase_at_Y, predict_phase_custom_wavelvet, plot_average_epochs, plot_compare_average_epochs,plot_highlighted_psd_prestim


#%%


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description')

# Add arguments
parser.add_argument('--duration', type=int, default=1)
parser.add_argument('--source_ch', type=list, default=['Afz'])
parser.add_argument('--freqs', type=list, default=['6'])
parser.add_argument('--n_cycles', type=int, default=2)
parser.add_argument('--prestim_time', type=int, default=5) # for phase estimation (ms)
parser.add_argument('--peak_width', type=int, default=80,help='±peak_width/2 to both direction') #±peak_width/2 to both direction
parser.add_argument('--csd_transform', type=bool, default=True) 
parser.add_argument('--auto_reject_rest', type=bool, default=True) 

# Parse the command-line arguments
args = parser.parse_args('')


sub_list = [1,2,3,4,5,6,8,9,10,11]
#sub_list = [8]
# Define file paths and subject list
current_path = pathlib.Path().absolute()
save_path = current_path / 'prepro'

# Read preprocessed data


tep_files = [save_path /get_sub_str(sub_num) / f'{get_sub_str(sub_num)}_prepro-epo.fif' for sub_num in sub_list]


# Read the TFR data
group_tfr = []
for fname,sub_num in zip(tep_files,sub_list):
    epochs_tep = mne.read_epochs(fname)

# %%
