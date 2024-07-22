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
# %%


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

# %%

below_median_tfrs = [tfr_dict[sub].average() for sub in below_median['Subject Number']]
above_median_tfrs = [tfr_dict[sub].average() for sub in above_median['Subject Number']]

#%%

grand_below =  mne.grand_average(below_median_tfrs)
grand_above =  mne.grand_average(above_median_tfrs)
#%%

channel_name = 'PO3'
mode = 'mean'
data_below = grand_below.data[grand_below.ch_names.index(channel_name)]
data_above = grand_above.data[grand_above.ch_names.index(channel_name)]


# Determine the common color scale (vmin and vmax)
vmin = min(data_below.min(), data_above.min())
vmax = max(data_below.max(), data_above.max())

# Plot the TFRs side by side with the same color scale
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot TFR for the 'below' condition
grand_below.plot([grand_below.ch_names.index(channel_name)], axes=axes[0], baseline=(None, 0), mode='mean', vmin=vmin, vmax=vmax)

# Plot TFR for the 'above' condition
grand_above.plot([grand_above.ch_names.index(channel_name)], axes=axes[1], baseline=(None, 0), mode='mean', vmin=vmin, vmax=vmax)

axes[0].set_title(f'Low VVIQ {channel_name}')

axes[1].set_title(f'High VVIQ {channel_name}')
# Adjust layout
plt.tight_layout()

plt.show()

