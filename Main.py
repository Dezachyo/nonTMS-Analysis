#%% Imports

import pandas as pd
import PyQt5
import os
import math
import numpy as np
import pathlib
import mne
from mne_icalabel import label_components
import matplotlib.pyplot as plt
from subfunctions import load_vhdr, get_behav
import logging

import autoreject
from IPython.display import display,HTML
from IPython import get_ipython
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
        if event.startswith('baseline'):
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
    
    #return HTML(f"<font color='{colors[color]}'>{text}</font>")
    display(HTML(f"<font color='{colors[color]}'>{text}</font>"))
import pandas as pd
import matplotlib.pyplot as plt

def plot_autoreject_per_event(epochs, epochs_ar):
    # Initialize lists to store data
    keys = []
    droped = []
    kept = []

    # Iterate over the items in epochs.event_id
    for key, value in epochs.event_id.items():
        keys.append(key)
        total = len(epochs[key])
        accept = len(epochs_ar[key])
        rejected = total - accept

        droped.append(rejected)
        kept.append(accept)

    # Create a DataFrame
    data = {
        'Key': keys,
        'droped': droped,
        'kept': kept
    }
    df = pd.DataFrame(data)

    # Set the width of the bars
    bar_width = 0.35

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Plotting the stacked bars with custom colors
    plt.bar(df['Key'], df['kept'], bottom=df['droped'], label='Kept', color='lightgrey', width=bar_width)
    plt.bar(df['Key'], df['droped'], label='Dropped', color='lightcoral', width=bar_width)

    # Add labels and title
    plt.xlabel('Event')
    plt.ylabel('Count')
    plt.title('Autoreject Drops Per Event')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
    return df



# %% Automatic prepro pipeline

# Load data -> resample -> epoching -> Highpass (0.1Hz) -> ica? -> autoreject -> reref

# Set args for prepro

sub_num = 11

prepro_args = {'resample' : 1000,
               'tmin' : -0.6,
               'tmax' : 1,
               'baseline' : None,
               'drop ica' : ['eye blink', 'muscle artifact','channel noise']
             }



event_dict = {'object': 31,
              'feature': 32,
              'scene': 33,
              'binding': 51,
              'baseline': 61,
              'retrieval' : 41,
              }

#Set Paths

current_path = pathlib.Path().absolute()
zeros = (3- len(str(sub_num)))*'0' 
sub_str = f'sub_'+zeros+f'{sub_num}'
data_path = current_path.parent / 'Episodic-TEP-Analysis' /'Data'/ sub_str
results_path = current_path/'prepro'/sub_str  
results_path.mkdir(parents=True, exist_ok=True) # Make Directory to save results in

vhdr_fname = data_path  /'EEG'/ f'{sub_str}_task_TEP.vhdr' 
raw = load_vhdr(vhdr_fname,load_montage=True, EOG_ch=False)
df_behav = get_behav(sub_num)


#init log file
log_file_name = "log.log"

# Construct the full path to the log file
log_file_path = results_path / log_file_name
# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Example log message
logging.info('Raw loaded')

raw = raw.resample(prepro_args['resample'])
logging.info(f'Resampled to {prepro_args["resample"]} Hz')

# Drop EMG channels if present 
if 'EMG' in raw.ch_names:       
    raw.drop_channels(['EMG'])


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
    message = f"Number of detected trials (df_behav):{num_trials}"
    cprint(message, 'green')
    logging.info(message)
else:
    message = f"Number of detected trials (df_behav):{num_trials}"
    cprint(message, 'red')
    logging.info(message)

#add behavioral measurments to metadata
df_behav['trial_num'] =  range(1,num_trials+1)# add a trial num column 
metadata_combined = pd.merge(metadata_with_trial_num, df_behav, on='trial_num', how='left')
#TODO use data from df_behav to add trial num for ret events 


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
logging.info(f'{len(epochs)} Epoches created')


epochs = mne.set_eeg_referance(epochs)
#%% ICA

#Fit ICA for >1Hz signal.

epochs_for_ica_fit = epochs.copy().filter(l_freq=1, h_freq=None)


# Fit ICA 
#TODO Make sure ICA is fitted right for ica_label
ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True),random_state = 100)
ica.fit(epochs_for_ica_fit)


ic_labels = label_components(epochs_for_ica_fit, ica, method="iclabel")
labels = ic_labels["labels"]
exclude_idx = [
    idx for idx, label in enumerate(labels) if label in prepro_args['drop ica']
]
print(f"Excluding these ICA components: {exclude_idx}")

del(epochs_for_ica_fit)

#Applay ICA for >0.1Hz signal 

epochs_pre_ica = epochs.copy()
# plot overly
ica.plot_overlay(epochs_pre_ica.average(), exclude=exclude_idx)
plt.title(f'{len(exclude_idx)} Components Rejected  \n subject {sub_num}')
plt.savefig(results_path/'ICA_Evoked_Overlay.png')

ica.plot_components(exclude_idx)
plt.savefig(results_path/'ICA_Exclude_Comp.png')

ica.apply(epochs, exclude=exclude_idx)
msg = f"ICA comp rejected {[(idx,label) for idx, label in enumerate(labels) if label in prepro_args['drop ica']]}"
cprint(msg,'green')
logging.info(msg)


plot_kwargs = {
    'n_epochs': 12,
    'n_channels': 15,
    'scalings': dict(eeg=60e-6)
}

#epochs_pre_ica.plot(**plot_kwargs,title = f"Pre ICA")
#epochs.plot(**plot_kwargs, title = f"Post ICA rejection ({len(exclude_idx)} components automaticly excluded)" )

del(epochs_pre_ica)
# %% Autoreject
logging.info('Running Autoreject')

ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:100])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

fig = epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

#TODO Save fig to directory (error with incerting Path fname)
save_path = results_path / 'Autoreject_Bad_Epochjs.png'
fig.grab().save(str(save_path))


reject_log.plot('horizontal')
plt.savefig(results_path/'Autoreject_Reject_LOG')

df_ar_event_drop = plot_autoreject_per_event(epochs, epochs_ar)
plt.savefig(results_path/'Autoreject_Reject_Per_Event')

# Update epochs with autoreject results
epochs = epochs_ar
msg = f'Autoreject completed, {len(reject_log.bad_epochs)} epochs rejected'
logging.info(df_ar_event_drop[['droped','Key']])
logging.info(msg)
logging.info('Autoreject Completed')

# %%

