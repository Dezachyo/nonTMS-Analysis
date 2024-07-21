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


for sub_num in tqdm(sub_list):

    sub_str = get_sub_str(sub_num)
    current_path = pathlib.Path().absolute()
    results_path = current_path/'TFR'/sub_str 
    fname_tfr = f'{sub_str}-tfr.h5'

    tfr = mne.time_frequency.read_tfrs(results_path/fname_tfr)