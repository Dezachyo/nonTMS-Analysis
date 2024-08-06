#%% Imports
import pandas as pd
import PyQt5
import math
import numpy as np
import pathlib
import mne
from mne.stats import permutation_cluster_test,combine_adjacency,spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind
import mne.stats
import scipy.stats as stats
from tqdm import tqdm
from subfunctions import get_sub_str, cprint

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


def plot_tfr_conditions(tfr, channel_name, baseline=(-0.4, 0), baseline_mode='percent'):
    """
    Plots the TFRs for each condition in the given channel.

    Parameters:
    - tfr: MNE Time-Frequency Representation (TFR) object
    - channel_name: The name of the channel to plot
    - baseline: Baseline time window (tuple, start and end)
    - baseline_mode: Baseline correction mode (e.g., 'percent', 'zscore')
    """
    # Extract data for the given channel
    channel_index = tfr.ch_names.index(channel_name)

    # Identify unique conditions from the metadata using the correct column name 'event_name'
    conditions = tfr.metadata['event_name'].unique()

    # Initialize vmin and vmax
    vmin = float('inf')
    vmax = float('-inf')

    # First pass to apply baseline correction and determine global vmin and vmax
    for condition in conditions:
        condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
        condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
        tfr_condition = tfr[condition].copy().pick([channel_name])
        tfr_condition.apply_baseline(baseline, mode=baseline_mode)
        data = tfr_condition.data.mean(axis=0)
        vmin = min(vmin, data.min())
        vmax = max(vmax, data.max())

    # Create subplots
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))

    # Plot each TFR in a subplot with the same color scale
    for i, condition in enumerate(conditions):
        condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
        condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
        tfr_condition = tfr[condition].copy().pick([channel_name])
        tfr_condition.apply_baseline(baseline, mode=baseline_mode)
        tfr_condition.average().plot(axes=axes[i], vmin=vmin, vmax=vmax, show=False)
        axes[i].set_title(condition)
        axes[i].set_xlabel('Time (s)')
        if i != 0:
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])  # Remove y-tick labels for all but the first subplot
        else:
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_yticks(tfr_condition.freqs)
            axes[i].set_yticklabels([f'{freq:.1f}' for freq in tfr_condition.freqs])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    return fig

# %%

sub_list = [1,2,3,4,5,6,8,9,10,11]
#sub_list = [8]
# Define file paths and subject list
current_path = pathlib.Path().absolute()
results_path = current_path / 'TFR'

tfr_files = [results_path /get_sub_str(sub_num) /f'{get_sub_str(sub_num)}-tfr.h5' for sub_num in sub_list]

# Read the TFR data
group_tfr = []
for fname,sub_num in zip(tfr_files,sub_list):
    tfr = mne.time_frequency.read_tfrs(fname)
    #plot_tfr_conditions(tfr[0], 'POz')
    fig = plot_tfr_conditions(tfr[0], 'POz',baseline_mode='logratio')
    fig.suptitle(f'subjct {sub_num}')

    conditions = tfr[0].metadata['event_name'].unique()
    tfr_avg = {condition: tfr[0][condition].average() for condition in conditions}
    group_tfr.append(tfr_avg)
    del(tfr)

#%%



#%% Permutation test

n_permutations = 1000
p_value_threshold = 0.05 # Cluster correction 

a = [tfr['binding'].data.transpose(1,2,0) for tfr in group_tfr]
b = [tfr['object'].data.transpose(1,2,0) for tfr in group_tfr]

X = [np.stack(a),np.stack(b)]

p = 0.05 # For trial data I used 0.001
df = len(a) - 2
t_threshold = stats.distributions.t.ppf(1 - p / 2, df=df)
# our data at each observation is of shape frequencies × times × channels)

tfr_info =  group_tfr[0]['object'].info
tfr_times = group_tfr[0]['object'].times
tfr_freqs = group_tfr[0]['object'].freqs

adjacency, ch_names = find_ch_adjacency(tfr_info, ch_type= None)

tfr_adjacency = combine_adjacency(len(tfr_freqs), len(tfr_times), adjacency)

# run cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(
    X,
    n_permutations=n_permutations,
    threshold=t_threshold,
    tail=0,
    n_jobs=None,
    buffer_size=None,
    adjacency=tfr_adjacency,
    #stat_fun= ttest_rel_nop
    stat_fun= mne.stats.ttest_ind_no_p
)

T_obs, clusters, p_values, H0 = cluster_stats
good_cluster_inds = np.where(p_values < p_value_threshold)[0]
