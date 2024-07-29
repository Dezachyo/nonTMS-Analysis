#%% Imports
import pandas as pd
import PyQt5
import math
import numpy as np
import pathlib
import mne
from mne.time_frequency import tfr_morlet

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
subject_list = [1, 3]  # List of subjects or conditions
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
occ_picks = ['PO4','PO8','POz','O2','Oz','PO3','PO7','Pz','Oz']

for condition in conditions:
    grand = mne.grand_average([tfr_dict[condition] for tfr_dict in group_tfr])
    grand.plot_topo(baseline=(-0.4, 0),mode="logratio", title = condition)
    grand.plot(baseline=(-0.4, 0),mode="logratio", picks=occ_picks,combine='mean', title = condition + f'\n{occ_picks}')
    

#%%

#tfr_group = [mne.time_frequency.read_tfrs(fname)[0] for fname in tfr_files]
#tfr_dict =  dict(zip(sub_list, tfr_group))



tfr['binding'].average().plot_joint(
    baseline=(-0.4, 0.02), mode="mean", tmin=-0.5, tmax=2, timefreqs=[(0.2,6.5),(0.5, 10), (0.3, 10)]
)


# %%

below_median_tfrs = [tfr_dict[sub].average() for sub in below_median['Subject Number']]
above_median_tfrs = [tfr_dict[sub].average() for sub in above_median['Subject Number']]

#%%

grand_below =  mne.grand_average(below_median_tfrs)
grand_above =  mne.grand_average(above_median_tfrs)


#%% ==================== condition comparison (with mne ploting) =========


#%% ==================== condition comparison (no baseline) =========

tfr = l[0]
# Define the channel name
channel_name = 'POz'

# Extract data for the given channel
channel_index = tfr.ch_names.index(channel_name)

# Identify unique conditions from the metadata using the correct column name 'event_name'
conditions = tfr.metadata['event_name'].unique()

# Determine the common color scale (vmin and vmax)
vmin = float('inf')
vmax = float('-inf')

# First pass to determine global vmin and vmax
for condition in conditions:
    condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
    condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
    data = tfr.data[condition_indices, channel_index, :, :].mean(axis=0)
    vmin = min(vmin, data.min())
    vmax = max(vmax, data.max())

# Create subplots
fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))

# Plot each TFR in a subplot with the same color scale
for i, condition in enumerate(conditions):
    condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
    condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
    data = tfr.data[condition_indices, channel_index, :, :].mean(axis=0)
    im = axes[i].imshow(data, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, 
                        extent=[tfr.times.min(), tfr.times.max(), tfr.freqs.min(), tfr.freqs.max()])
    axes[i].set_title(condition)
    axes[i].set_xlabel('Time (s)')
    if i == 0:
        axes[i].set_ylabel('Frequency (Hz)')
    else:
        axes[i].set_yticklabels([])

# Add a shared colorbar, adjusting its position to avoid overlap
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Power (dB)')

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()