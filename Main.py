#%% Imports

import pandas as pd
import PyQt5
import os
import math
import numpy as np
import pathlib
import mne
import matplotlib.pyplot as plt
from subfunctions import load_vhdr, get_behav
from IPython import get_ipython
import autoreject
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'qt')


def add_trial_numbers(metadata):
    """For a metadata created from annotations, for multiple conditions

    Args:
        metadata (pd.Dataframe): created with mne.epochs.make_metadata.

    Returns:
        _type_: _description_
    """
    trial_num = 0
    trial_numbers = []

    for event in metadata['event_name']:
        if event.startswith('contrast'):
            trial_num += 1
        trial_numbers.append(trial_num)

    metadata['trial_num'] = trial_numbers
    metadata.loc[metadata['event_name'] == 'retrieval', 'trial_num'] = None
    return metadata, max(trial_numbers)


def cprint(text, color='white'):
    colors = {
        'red': '#FF0000',
        'green': '#00FF00',
        'white': '#FFFFFF'
    }
    
    if color not in colors:
        raise ValueError("Invalid color. Choose from 'red', 'green', or 'white'.")
    
    return HTML(f"<font color='{colors[color]}'>{text}</font>")

# %% Automatic prepro pipeline

# Load data -> resample -> epoching -> ica? -> autoreject -> reref

# Set args for prepro

prepro_args = {'resample' : 1000,
               'tmin' : -0.6,
               'tmax' : 1,
               'baseline' : None
             }

sub_num = 11

event_dict = {'object': 31,
              'feature': 32,
              'scene': 33,
              'binding': 51,
              'contrast': 61,
              'retrieval' : 41,
              }


current_path = pathlib.Path().absolute()
zeros = (3- len(str(sub_num)))*'0' 
sub_str = f'sub_'+zeros+f'{sub_num}'
data_path = current_path.parent / 'Episodic-TEP-Analysis' /'Data'/ sub_str

vhdr_fname = data_path  /'EEG'/ f'{sub_str}_task_TEP.vhdr' 
raw = load_vhdr(vhdr_fname,load_montage=True, EOG_ch=False)
df_behav = get_behav(sub_num)

raw = raw.resample(prepro_args['resample'])

events_from_annot,_ = mne.events_from_annotations(raw)
selected_events = [x for x in events_from_annot if x[2] in event_dict.values()]
# Just a trick to make it np.array for MNE
selected_events = np.array(selected_events).tolist()
selected_events = np.array(selected_events)

metadata, _, _ = mne.epochs.make_metadata(
    events=selected_events,
    event_id=event_dict,
    tmin=0,
    tmax=0,
    sfreq=raw.info["sfreq"],
)



metadata_with_trial_num, num_trials = add_trial_numbers(metadata)

if num_trials == 96:
    cprint(f"Number of detected trials:{num_trials}", 'green')
else:
    cprint(f"Number of detected trials:{num_trials}", 'red')
    


#add behavioral measurments to metadata
df_behav['trial_num'] =  range(1,num_trials+1) # add a trial num column 
metadata_combined = pd.merge(metadata_with_trial_num, df_behav, on='trial_num', how='left')



epochs = mne.Epochs(raw,
                    events = selected_events,
                    event_id = event_dict,
                    baseline = prepro_args['baseline'],
                    tmin=prepro_args['tmin'],
                    tmax=prepro_args['tmax'],
                    preload=True,
                    reject = None,
                    detrend=0,
                    metadata = metadata_combined
                    )
# %% Autoreject

ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:100])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal')