"""
Comodulogram
------------
This example creates an artificial signal with phase-amplitude coupling (PAC)
and computes comodulograms with several methods.

A comodulogram shows the estimated PAC metric on a grid of frequency bands.
"""

import numpy as np
import matplotlib.pyplot as plt

from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac

from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet

from mne.stats import permutation_cluster_test

from subfunctions import load_vhdr,get_sub_str
import mne
import pathlib

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


# Load EEG from my experimet 
sub_num = 6
sub_str = get_sub_str(sub_num)
current_path = pathlib.Path().absolute()
data_path = current_path / 'prepro' / sub_str

fname = f'{sub_str}_prepro-epo.fif'
save_path = data_path / fname
# Read preprocessed data
epochs_tep = mne.read_epochs(save_path)

resample_fs = None

if resample_fs is not None:
    epochs_tep.resample(resample_fs)   
else:
    resample_fs = epochs_tep.info['sfreq']
    

signal = epochs_tep['binding']._data[20,6,:]

###############################################################################
# Let's first create an artificial signal with PAC.

fs = resample_fs  # Hz 200 originaly
high_fq = 50.0  # Hz
low_fq = 5.0  # Hz
low_fq_width = 1.0  # Hz

n_points = 10000
noise_level = 0.4

signal_sim = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                      low_fq_width=low_fq_width, noise_level=noise_level,
                      random_state=0)

###############################################################################
# Then, let's define the range of low frequency, and the list of methods used

low_fq_range = np.linspace(1, 10, 50)
methods = [
    'ozkurt', 'canolty', 'tort', 'penny', 'vanwijk', 'duprelatour', 'colgin',
    'sigl', 'bispectrum'
]

###############################################################################
# To compute the comodulogram, we need to instanciate a `Comodulogram` object,
# then call the method `fit`. The method `plot` draws the results on the given
# subplot axes.

# Define the subplots where the comodulogram will be plotted
n_lines = 3
n_columns = int(np.ceil(len(methods) / float(n_lines)))
fig, axs = plt.subplots(
    n_lines, n_columns, figsize=(4 * n_columns, 3 * n_lines))
axs = axs.ravel()


# Compute the comodulograms and plot them
for ax, method in zip(axs, methods):
    print('%s... ' % (method, ))
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, method=method,
                             progress_bar=False)
    estimator.fit(signal)
    estimator.plot(titles=[REFERENCES[method]], axs=[ax])

plt.show()


#%% TensorPac

epochs_signal = epochs_tep['binding']._data[0:60,25,:]*10e5


###############################################################################
# Compute the single trial PAC on both datasets
###############################################################################
# once the datasets created, we can now extract the PAC, computed across
# time-points for each trials and across several phase and amplitude
# frequencies

# create the pac object. Use the Gaussian-Copula PAC
p = Pac(idpac=(2, 0, 0), f_pha='hres', f_amp='hres', dcomplex='wavelet')
# compute pac for both dataset
pac_1 = p.filterfit(resample_fs, epochs_signal, n_jobs=-1)

plt.figure()
p.comodulogram(pac_1)

###############################################################################
# Compute the single trial PAC on both datasets
###############################################################################
# once the datasets created, we can now extract the PAC, computed across
# time-points for each trials and across several phase and amplitude
# frequencies

# create the pac object. Use the Gaussian-Copula PAC
p = Pac(idpac=(2, 0, 0), f_pha='hres', f_amp='hres', dcomplex='wavelet')
# compute pac for both dataset
pac_1 = p.filterfit(resample_fs, epochs_signal, n_jobs=-1)

plt.figure()
p.comodulogram(pac_1)

from tensorpac.utils import BinAmplitude

plt.figure(figsize=(14, 5))

plt.subplot(121)
b_obj = BinAmplitude(epochs_signal, fs, f_pha=[9, 11], f_amp=[40, 70], n_jobs=1,
                     n_bins=18)
ax = b_obj.plot(color='red', alpha=.5, unit='deg')
plt.ylim(0, 180)
plt.title("Binned amplitude (phase=[9, 11])")

plt.subplot(122)
b_obj = BinAmplitude(epochs_signal, fs, f_pha=[4, 8], f_amp=[40, 70], n_jobs=1,
                     n_bins=18)
ax = b_obj.plot(color='blue', alpha=.5, unit='deg')
plt.ylim(0, 180)
plt.title("Binned amplitude (phase=[4, 8])")

b_obj.show()