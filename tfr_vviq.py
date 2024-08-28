#%% Imports
import pandas as pd
import PyQt5
import math
import numpy as np
import pathlib
import mne
from mne.time_frequency import tfr_morlet
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

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
# %%

sub_list = [1,2,3,4,5,6,8,9,10,11]

baseline = (-0.4 , -0.05)

# Define file paths and subject list
current_path = pathlib.Path().absolute()
results_path = current_path / 'TFR'

tfr_files = [results_path /get_sub_str(sub_num) /f'{get_sub_str(sub_num)}-tfr.h5' for sub_num in sub_list]

# Read the TFR data
group_tfr = []
for fname,sub_num in zip(tfr_files,sub_list):
    tfr = mne.time_frequency.read_tfrs(fname)
    #plot_tfr_conditions(tfr[0], 'POz')
    
    
    #fig = plot_tfr_conditions(tfr[0], 'POz',baseline_mode='logratio')
    #fig.suptitle(f'subjct {sub_num}')

    tfr[0].apply_baseline(mode="logratio", baseline=baseline)
    
    conditions = tfr[0].metadata['event_name'].unique()
    tfr_avg = {condition: tfr[0][condition].average() for condition in conditions}
    group_tfr.append(tfr_avg)
    del(tfr)
    
tfr_dict =  dict(zip(sub_list, group_tfr)) 
# %% Read viq

data_path = current_path.parent / 'Episodic-TEP-Analysis' / 'Data'
vvix_file_path = data_path / 'group_level' / 'Processed_VVIQ.csv'

# Load the CSV file
vvix_data = pd.read_csv(vvix_file_path)

# Convert 'Subject Number' to a categorical type with the specified order
vvix_data['Subject Number'] = pd.Categorical(vvix_data['Subject Number'], categories=sub_list, ordered=True)


# Assuming your DataFrame is named vvix_data
vvix_data_sorted = vvix_data[['Subject Number', 'VVIQ score']].sort_values('VVIQ score').dropna()

# Calculate the median of the 'VVIQ score' column
median_score = vvix_data_sorted['VVIQ score'].median()

# Get rows below the median
below_median = vvix_data_sorted[vvix_data_sorted['VVIQ score'] < median_score]

# Get rows above the median
above_median = vvix_data_sorted[vvix_data_sorted['VVIQ score'] > median_score]

# Include the rows that have the median score in the below_median DataFrame if desired
median_rows = vvix_data_sorted[vvix_data_sorted['VVIQ score'] == median_score]

# Combine below_median and median_rows if needed
below_median = pd.concat([below_median, median_rows])

# %% Show Results

save_fig = True

for condition in conditions:

    below_median_tfrs = [tfr_dict[sub][condition] for sub in below_median['Subject Number']]
    above_median_tfrs = [tfr_dict[sub][condition] for sub in above_median['Subject Number']]

    grand_below =  mne.grand_average(below_median_tfrs)
    grand_above =  mne.grand_average(above_median_tfrs)

    occ_picks = ['PO4','PO8','POz','O2','PO3','PO7','Pz','Oz']

    grand_below.pick(occ_picks)
    grand_above.pick(occ_picks)

    grand_below.data.mean(axis = 0)

    data_below = grand_below.data.mean(axis = 0)
    data_above = grand_above.data.mean(axis = 0)


    # Determine the common color scale (vmin and vmax)
    vmin = min(data_below.min(), data_above.min())
    vmax = max(data_below.max(), data_above.max())

    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    # Plot the TFRs side by side with the same color scale
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot TFR for the 'below' condition
    grand_below.plot(combine='mean',
                     axes=axes[0],
                     vmin=vmin,
                     vmax=vmax,
                     cnorm = cnorm
                     )

    # Plot TFR for the 'above' condition
    grand_above.plot(combine='mean',
                     axes=axes[1],
                     vmin=vmin,
                     vmax=vmax,
                     cnorm = cnorm
                     )

    fig.suptitle(f'{condition} \n {occ_picks}')

    axes[0].set_title(f'Low VVIQ')

    axes[1].set_title(f'High VVIQ')
    # Adjust layout
    plt.tight_layout()
    plt.show()
    if save_fig:
        fig_path = current_path / 'Figures'/'vviq_induced_ocill'
        fig.savefig(fig_path /f'Median Split_VVVIQ_{condition}_occsipital.png', dpi=300, bbox_inches='tight')


#%% ------------------------------- Same as above but all conditions in one figure -------------------



# Adjust the figure size to be taller and less wide, aiming for more square-like subplots
fig, axes = plt.subplots(len(conditions), 2, figsize=(8, 8 * len(conditions)))

# Define the range of time (x-axis) based on your data
time_range = np.linspace(-0.5, 1.0, 5)  # Adjust the number of ticks here (e.g., 5 ticks)

# Loop through each condition and plot the TFRs
for i, condition in enumerate(conditions):
    
    below_median_tfrs = [tfr_dict[sub][condition] for sub in below_median['Subject Number']]
    above_median_tfrs = [tfr_dict[sub][condition] for sub in above_median['Subject Number']]

    grand_below =  mne.grand_average(below_median_tfrs)
    grand_above =  mne.grand_average(above_median_tfrs)

    occ_picks = ['PO4','PO8','POz','O2','PO3','PO7','Pz','Oz']

    grand_below.pick(occ_picks)
    grand_above.pick(occ_picks)

    data_below = grand_below.data.mean(axis=0)
    data_above = grand_above.data.mean(axis=0)

    # Determine the common color scale (vmin and vmax)
    vmin = min(data_below.min(), data_above.min()) - 0.3
    vmax = max(data_below.max(), data_above.max()) + 0.3

    # Plot TFR for the 'below' condition
    grand_below.plot(combine='mean', axes=axes[i, 0], vmin=vmin, vmax=vmax,cnorm = cnorm, show=False)

    # Plot TFR for the 'above' condition
    grand_above.plot(combine='mean', axes=axes[i, 1], vmin=vmin, vmax=vmax,cnorm = cnorm, show=False)

    # Set titles for the plots
    axes[i, 0].set_title(f'{condition} - Low VVIQ')
    axes[i, 1].set_title(f'{condition} - High VVIQ')

    # Set x-ticks manually to avoid overlap
    axes[i, 0].set_xticks(time_range)
    axes[i, 1].set_xticks(time_range)

# Adjust layout
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Add an overall title to the figure
fig.suptitle('TFR Comparison Across Conditions', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




# %% Could have been great way to read tfrs but there is no memory for all in once


sub_list = [1,3,4,5,6,8,9,10,11]
# Define file paths and subject list
current_path = pathlib.Path().absolute()
results_path = current_path / 'TFR'
subject_list = [1, 3]  # List of subjects or conditions
tfr_files = [results_path /get_sub_str(sub_num) /f'{get_sub_str(sub_num)}-tfr.h5' for sub_num in sub_list]

# Read the TFR data
tfr_group = [mne.time_frequency.read_tfrs(fname)[0] for fname in tfr_files]

tfr_dict =  dict(zip(sub_list, tfr_group))

tfr_group[-1]['binding'].average().plot_joint(
    baseline=(-0.4, 0.02), mode="mean", tmin=-0.5, tmax=2, timefreqs=[(0.2,6.5),(0.5, 10), (0.3, 10)]
)