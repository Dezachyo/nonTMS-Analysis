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

#%%

sub_list = [1,2,3,4,5,6,8,9,10,11]

freqs = np.logspace(*np.log10([3, 40]), num=25)
n_cycles = np.linspace(2.6, 2.6 + 0.2 * (30 - 1), len(freqs))
decim = 4

#freqs = np.arange(4, 30, 2)  # define frequencies of interest
#n_cycles = 2
#n_cycles = freqs / 2.0  # different number of cycle per frequency

for sub_num in tqdm(sub_list):

    sub_str = get_sub_str(sub_num)
    current_path = pathlib.Path().absolute()
    data_path = current_path / 'prepro' / sub_str

    fname = f'{sub_str}_prepro-epo.fif'
    save_path = data_path / fname
    # Read preprocessed data
    epochs_tep = mne.read_epochs(save_path)

    tfr = tfr_morlet(
        epochs_tep,
        freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
    )

    tfr["binding"].average().plot_topo(baseline=(-0.45, -0.1),
                                       mode="percent",
                                       title=f"Subject {sub_num} \n Average power (binding)")

    # Save trf to disc
    results_path = current_path/'TFR'/sub_str  
    results_path.mkdir(parents=True, exist_ok=True) # Make Directory to save results in

    fname_tfr = f'{sub_str}-tfr.h5'

    tfr.save(results_path/fname_tfr ,overwrite=True)
    cprint(f'Subject {sub_num} TFR Saved to disc ','green')




#%% Compare TFR for each subjct as well as grand average 

def save_tfr_ch_figs(tfr_avg,sub_path,condition):
    #TODO Close ch fig after saving
    """create a folder for channels figs

    Args:
        tfr_avg (mne): AVG TFR
        sub_path (_type_): Pathlib (Figuers/sub_num)
        condition (str): folder name
    """
    path = fig_path/condition
    if not path.is_dir():
        path.mkdir(parents=True)
    folder_name = condition
    tfr_avg.plot_topo(title = f'subject {sub_num}' + title)
    plt.savefig(fig_path/folder_name/condition)
    for ch in tfr_avg.ch_names:
        tfr_avg.plot(picks = ch, title = f'{ch}')
        plt.savefig(fig_path/folder_name/f'{ch}')
        plt.close()
    
     

tfr_both = mne.time_frequency.read_tfrs(save_path)