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

from subfunctions import get_sub_str

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#%%
sub_num = 8

sub_str = get_sub_str(sub_num)
current_path = pathlib.Path().absolute()
data_path = current_path / 'prepro' / sub_str

fname = f'{sub_str}_prepro-epo.fif'
save_path = data_path / fname

epochs_tep = mne.read_epochs(save_path)



decim = 4
#freqs = np.arange(4, 30, 2)  # define frequencies of interest
#n_cycles = 2

freqs = np.logspace(*np.log10([3, 40]), num=25)
n_cycles = np.linspace(2.6, 2.6 + 0.2 * (30 - 1), len(freqs))

#n_cycles = freqs / 2.0  # different number of cycle per frequency


tfr_all = tfr_morlet(
    epochs_tep,
    freqs,
    n_cycles=n_cycles,
    
    return_itc=False,
    average=False,
)


tfr_all["binding"].average().plot_topo(baseline=(-0.45, -0.1), mode="percent", title="Average power")
# %%
