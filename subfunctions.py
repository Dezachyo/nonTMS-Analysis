#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:45:14 2022

@author: ordez
"""
import mne
from fooof import FOOOF
#from mne.time_frequency import psd_welch
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test
import pandas as pd
import pathlib
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

from IPython.display import display,HTML
from IPython import get_ipython

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

def get_sub_str(sub_num):
    zeros = (3- len(str(sub_num)))*'0' 
    sub_str = f'sub_'+zeros+f'{sub_num}'
    return sub_str

def get_behav(sub_num):
    
    current_path = pathlib.Path().absolute()
    
    zeros = (3- len(str(sub_num)))*'0' 
    sub_str = f'sub_'+zeros+f'{sub_num}'
    csv_path = current_path.parent / 'Episodic-TEP-Analysis' /'Data'/ sub_str/'Task'/ f'sub{sub_num}.csv'
    df = pd.read_csv(csv_path)

 
    df = df.dropna()
    trials_num = df.shape[0]
    print( f'Number of valid trials: {trials_num}')

    # A dict of labels for recall column
    recall_lab = {2 : 'All', 1:'One' ,0:'None'}

    # make a new column with the number if items recalled:
    df['recall'] = df.apply(lambda x : recall_lab[2] if x['feat_mem'] == x['scene_mem'] and x['feat_mem'] == 'target' else '', axis=1)
    df['recall'] = df.apply(lambda x : recall_lab[1] if x['feat_mem'] != x['scene_mem'] and (x['feat_mem'] == 'target' or x['scene_mem']=='target') else x['recall']  , axis=1)
    df['recall']= df['recall'].replace('', recall_lab[0])    

    # Get thier idx
    #correct =  df.index[(df.feat_mem == 'target')] & df.index[(df.scene_mem == 'target')] .tolist()
    #correct_certain= df.index[(df.feat_mem == 'target')] & df.index[(df.scene_mem == 'target')] & df.index[(df['con_score'] == 'certain')]
    count_feat_mem = df.feat_mem.value_counts()

    feat_mem = count_feat_mem.target /(count_feat_mem.target + count_feat_mem.lure)
    return df


def load_vhdr( vdhr_fname , load_montage = True,EOG_ch = True  ):
    """" Doc """
    
    
    raw = mne.io.read_raw_brainvision(vdhr_fname,  verbose=False)
    
    raw.load_data()
    
    # set channel type for EMG and EOG
    raw.set_channel_types({"EMG": 'emg'})
    
    if 'HEGOC' in  raw.ch_names:
        raw.rename_channels({'HEGOC':'HEOG'})
        
    if EOG_ch == True:  
        raw.set_channel_types({"HEOG": 'eog',"VEOG": 'eog'})
        
    # Set montage
    montage = mne.channels.make_standard_montage('easycap-M1') # 
    
    # Find AFz in the montage and change it so it matches the true label "Afz" (f instead of F)
    i = montage.ch_names.index("AFz")
    montage.ch_names[i] = "Afz"
    
    raw.set_montage(montage, verbose=False)
            
    return raw


def fooof_channel_mean_welch (epochs,tmin,tmax,ch,fmin= 1, fmax = 60):
    """
    

    Parameters
    ----------
    epochs : mne.Epochs
        DESCRIPTION.
    tmin : int
        DESCRIPTION.
    tmax : int
        DESCRIPTION.
    ch : list
        a list of channels names ['Fz'].

    Returns
    -------
    Report.
    Plot.

    """
    
    sfreq=  epochs.info['sfreq']
    window = (tmax - tmin) # welch window in Sec
    n_fft = int(window*sfreq)
    # low n_fft results in less freq, need to be adjusted to the number of sampels or something
    epochs = epochs.copy() # make a copy to prevent modification to original instance

    epochs.pick_channels(ch) # Select only the Fz channel 

    spectra_ep , freqs = psd_welch(epochs,average = 'mean',n_fft=n_fft,fmin=1, fmax=60)

    #spectra_ch_mean = spectra_ep.mean(axis = 0)
    #spectra_ch_mean= spectra_ch_mean.mean(axis=0)

    #spectra_ch_mean =  np.squeeze(spectra_ch_mean)

    spectra_ch_mean =  np.squeeze(spectra_ep.mean(axis = (0,1))) # instead of the three lines above (avarage over channels and epochs)

    fig, ax = plt.subplots()
    ax.plot(freqs,spectra_ch_mean)

    # Initialize FOOOF object
    fm = FOOOF()

    # Define frequency range across which to model the spectrum
    freq_range = [2, 60]

    # Model the power spectrum with FOOOF, and print out a report
    fm.report(freqs, spectra_ch_mean, freq_range)
    plt.title(f'Channels ={ch} N = {epochs.events.shape[0]} (Welch Method)')
    
    return fm
def fooof_channel_mean (psd,ch,fmin= 1, fmax = 60,plot = True):
    """
    

    Parameters
    ----------
    psd : mne.Power Spectrum
        DESCRIPTION.
    tmin : int
        DESCRIPTION.
    tmax : int
        DESCRIPTION.
    ch : list
        a list of channels names ['Fz'].

    Returns
    -------
    Report.
    Plot.

    """
    
    sfreq=  psd.info['sfreq']
    psd = psd.copy() # make a copy to prevent modification to original instance

    psd.pick_channels(ch) # Select only the Fz channel 
    
    # Initialize FOOOF object
    fm = FOOOF()

    # Define frequency range across which to model the spectrum
    freq_range = [fmin, fmax]

    # Model the power spectrum with FOOOF, and print out a report
    fm.fit(psd.freqs, np.mean(psd,axis=(0,1)), freq_range)
    if plot == True:
        fm.plot()
        plt.title(f'Channels ={ch} N = {psd.get_data().shape[0]} (PSD)')
    
    return fm



    
def peak_params_select(fm_peak_params ,fmin,fmax):
    return fm_peak_params[np.logical_and(fm_peak_params[:,0] > fmin, fm_peak_params[:,0] < fmax)]


def epochs_peak(psd, fmin,fmax,fooof_plot = False):
    # psd -> Power Spectrum (MNE), if multichannel, mean across channels will be applied
    

    CF =[]
    PW =[]
    BW =[]

    # sub 20 first 3 epchos with theta
    # psd.get_data() -> np.array [epo,chan,power resulotion]

    for i,epo in enumerate(psd.get_data()):
        print(np.mean(epo,axis=0).shape)
        fm = FOOOF()

        # Define frequency range across which to model the spectrum
        freq_range = [1, 60]

        # Model the power spectrum with FOOOF, and print out a report
        fm.report(psd.freqs,np.mean(epo,axis=0), freq_range) # across Channels 
        plt.title(f'Epoch Number {i}')
        print (i)

        theta_peak = peak_params_select(fm.peak_params_, fmin,fmax)
        if theta_peak.shape[0] > 0:
            max_index = np.argmax(theta_peak[:, 0])
            CF.append(theta_peak[max_index][0])
            PW.append(theta_peak[max_index][1])
            BW.append(theta_peak[max_index][2])
        else:
            CF.append(None)
            PW.append(None)
            BW.append(None)       


    df_peaks = pd.DataFrame(list(zip(CF,PW,BW)), columns  = ['CF','PW','BW'])
    if fooof_plot == True:
        plt.show()
    return df_peaks
    
def hjorth_filter(epochs,h_filter =  {'Fz':1,'AF3':-0.25,'AF4':-0.25,'FC1':-0.25,'FC3':-0.25} ):
    """
    

    Parameters
    ----------
    epochs : mne.Epochs
        DESCRIPTION.
    
    Returns
    -------
    MNE.Epochs instance that only has 1 channel, that is the hjorth filter channel of Fz. Should be later rewritten to enable any channel and weights as an input

    """
    temp_epochs = epochs.copy()    
    ch_list = temp_epochs.ch_names.copy()
    
    # make sure that the required electrodes are in the dataset
    
    for ch in h_filter.keys():
        if ch not in ch_list:
            print(f'{ch} is not in dataset')
            return 
    
    # Get a list of non hjorth to remove
    for ch in h_filter.keys(): ch_list.remove(ch)
    # Drop all channels but hjorth channels
    temp_epochs.drop_channels(ch_list)
    # find channels indx
    i_ch = [temp_epochs.ch_names.index(key) for key in h_filter]
    hjorth = list(h_filter.keys())[0]
    group = dict(hjorth = i_ch)

    method = lambda data: data[:,0,:] - 0.25*data[:,1,:] -0.25*data[:,2,:] -0.25*data[:,3,:] -0.25*data[:,4,:]

    epochs_hjorth= mne.channels.combine_channels(temp_epochs,groups=group,method=method)
    return epochs_hjorth

def plot_SNR(fm,axis,title: str,fmin = 4, fmax = 8,plot = True):
    """
    Plot Fooof object with theta SNR to a axes
    
    Return: np.array([CF, SNR])
    
    """
    try: 
            theta_peak_param = fm.get_params('peak_params')[np.logical_and(fm.get_params('peak_params')[:,0] > fmin, fm.get_params('peak_params')[:,0] < fmax)]
    except IndexError:
            print(f'Params are {fm.get_params("peak_params")}, Returning 0 as peak')
            theta_peak_param = np.array([])
    if theta_peak_param.shape[0] == 0: # No Theta peak recognized:
        theta_peak_CF = 0
        theta_peak_SNR = 0
    else:   # Take the first Theta peak 
        theta_peak_CF = np.round(theta_peak_param[0,0],2)
        theta_peak_SNR = np.round(theta_peak_param[0,1] * 10,2)
        
    if plot == False:
        return theta_peak_SNR
            
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    try:
            fm.plot(ax = axis,plot_peaks=  'dot')
    except IndexError:
            fm.plot(ax = axis)
    axis.set_title(title, fontsize = 15)
    axis.set_xlabel('Frequency',fontsize = 12)
    axis.set_xlabel('Power',fontsize = 12)
    
    axis.text(0.45, 0.8, f'SNR ={theta_peak_SNR} dB\n Freq = {theta_peak_CF} HZ', transform=axis.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
def get_SNR(fm ,fmin = 4, fmax = 8):
    """
    
    Return: np.array([CF, SNR])
    
    """
    try: 
            theta_peak_param = fm.get_params('peak_params')[np.logical_and(fm.get_params('peak_params')[:,0] > fmin, fm.get_params('peak_params')[:,0] < fmax)]
    except IndexError:
            print(f'Params are {fm.get_params("peak_params")}, Returning 0 as peak')
            return 0
    if theta_peak_param.shape[0] == 0: # No Theta peak recognized:
        theta_peak_CF = 0
        theta_peak_SNR = 0
    else:   # Take the first Theta peak 
        theta_peak_CF = np.round(theta_peak_param[0,0],2)
        theta_peak_SNR = np.round(theta_peak_param[0,1] * 10,2)
        
    return theta_peak_SNR
            

def find_gfp_peaks(evoked, width =3,distance =15):
    #TODO Increasing levels of width and distance with time
    """
    Finds global field power peaks in an evoked object.

    Args:
        evoked: An evoked object.
        width: The width of the peak detection window in samples.
        distance: The minimum distance between peaks in samples.

    Returns:
        A np.array of peak times in seconds.
        A np.array of peak width in seconds.
    """
    # Find global field power peaks
    gfp = evoked.data.std(axis=0, ddof=0)
    peaks_pnt, _ = find_peaks(gfp,width = width, distance=distance)
    results_half = peak_widths(gfp, peaks_pnt, rel_height=0.5)
    results_half[0]  # widths
    
    
    peak_times = peaks_pnt/evoked.info['sfreq'] + evoked.tmin # time in (Sec)
    mask = np.logical_and(peak_times > 0, peak_times < 0.4) # Choose relevent peaks
    
    peak_times = peak_times[mask] # Select time 0-400 (Ms)
    peak_width = results_half[0][mask]/evoked.info['sfreq'] # In times
    print(f'Found {np.sum([np.logical_and(peak_times > 0, peak_times < 0.4)])} Peaks')
    return peak_times, peak_width
def peaks_to_time_windows(gfp_peaks,peaks_width,const_width = False,width = 0.01):
  """
  Returns a tuple that contains two variables for each value in the peaks array, tmin and tmax around the peak.

  Args:
    gfp_peaks: The np.array of peaks time (Sec) to iterate over.
    width: int. around the peak (Sec)
  Returns:
    A tuple of tuples, where each inner tuple contains two variables for the corresponding value in the np.array. tmin and tmax
  """

  time_windows1 = []
  for time,width in zip(gfp_peaks,peaks_width):

    if const_width:
        width = 0.01
        tmin = time - (width/2)
        tmax = time + (width/2)
        tmin = round(tmin,3)
        tmax = round(tmax,3)
        time_windows1.append((tmin, tmax))
    else:
        tmin = time - (width/2)
        tmax = time + (width/2)
        tmin = round(tmin,3)
        tmax = round(tmax,3)
        time_windows1.append((tmin, tmax))
        

  return tuple(time_windows1)



def create_phase_epochs_morlet(epochs_tep, freq, n_cycles):
    from mne.time_frequency import tfr_morlet
    """ Make an mne.Epochs represanting phase angles (rad) by morletwavelet transform (only works with one freq)

    Args:
        epochs_tep (mne.Epochs): _description_
        freq (float): One freq for morletwavelet
        n_cycels (float): number of cycles for morletwavelet

    Returns:
        _mne.Epochs_: phase angles
        _mne.Power : power inst resulting from morletwavelet transformation
    """
    
    power = tfr_morlet(epochs_tep, freqs=freq, n_cycles=n_cycles, return_itc=False,average = False,output = 'complex')

    np_angle = np.angle(power)
    phase_epochs_morlet = mne.EpochsArray(np.squeeze(np_angle), epochs_tep.info,tmin = epochs_tep.tmin,metadata = epochs_tep.metadata)
    #phase_epochs_morlet.plot(scalings=dict(eeg=4), n_epochs = 5, n_channels = 3)
    return phase_epochs_morlet, power

def plot_prestim_phase_dist_to_ax(degrees,ax):

    bin_size = 20
    a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

    #fig = plt.figure(figsize=(10,8))
    #ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.show()

def plot_peak_variation(df_metadata,args):
    # Assign colors to peak categories
    peak_colors = {'positive': 'tomato', 'negative': 'dodgerblue', 'neutral': 'gray'}

    # Create a scatter plot
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    
    
    for index, row in df_metadata.iterrows():
        # Calculate a random offset for x-coordinates to separate points
        x_offset = np.random.uniform(-0.05, 0.05)  # Smaller offset for better separation

        # Plot scatter points
        ax1.scatter(row['trial_num'] + x_offset, row['peak'], c=peak_colors[row['peak']], marker='o', s=50)

    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Peak')
    ax1.set_title('Peak Color Variation Over Time')
    plt.grid(True)

    value_counts = df_metadata['peak'].value_counts()
    ax2 = value_counts.plot(kind='bar', color=[peak_colors[val] for val in value_counts.index])
    ax2.set_title('Number of pulses')
    
    ax3 = fig.add_subplot(133, projection='polar')

    degrees = df_metadata['prestim_phase_deg']
    plot_prestim_phase_dist_to_ax(degrees,ax3)
    
    shaded_width = args.peak_width
    # Add a shaded area around 0 degrees
    ax3.fill_betweenx([0, ax3.get_ylim()[1]], np.deg2rad(-shaded_width/2), np.deg2rad(shaded_width/2), color='lightblue', alpha=0.5)

    # Add a shaded area around 180 degrees
    ax3.fill_betweenx([0, ax3.get_ylim()[1]], np.deg2rad(180 - shaded_width/2), np.deg2rad(180 + shaded_width/2), color='lightblue', alpha=0.5)

    
    # Adjust layout to avoid cutting off labels
    plt.tight_layout()

    return fig


def add_prestim_col (epochs_tep,phase_epochs,prestim_time,source_ch):
    """Add pre-stimulation phase estimation to Epochs metadata (in-plase)

    Args:
        epochs_tep (mne.Epochs): To be modified
        phase_epochs (mne.Epochs): epochs_tep transformation contains phase values 
        prestim_time (int: Pre-stimulation time to compure phase (ms positive)
        source_ch (list): one channel to compure phase

    Returns:
        mne.Epochs metadata modified
    """


    pulse_time =  -1*phase_epochs.tmin*1000 # Time (ms) of pulse relative to epoch start 
    phase_time =  pulse_time - prestim_time # Time (ms) to compute phase
    phase_sample = (phase_time/1000) * phase_epochs.info['sfreq']

    presim_phase =  phase_epochs.copy().pick_channels(source_ch).get_data()[:,0,int(phase_sample)]

    # Degrees: positive peak = 0 negative peak = 180
    # Radians: positibe peak = 0 negative peak = pi/-pi

    degrees = np.degrees(presim_phase) % 360 # transform -180-0 to 180-360
    radians = presim_phase

    # Assing to metadata 
    epochs_tep.metadata['prestim_phase_deg'] = degrees
    epochs_tep.metadata['prestim_phase_rad'] = radians
    
    print (f"Pulse time =  {pulse_time} ms | Estimating phase at {phase_time} ms")
    
    return epochs_tep

def add_peak_col(epochs_tep,peak_width):
    """Add pre-stimulation peak (negative,positive,neutral) to Epochs metadata (in-place)

    Args:
        epochs_tep (mne.Epochs): To be modified 
        peak_width (int): ±peak_width/2 to both direction

    Returns:
        mne.Epochs metadata modified
    """


    positive_peak_range = (360 - peak_width/2 , 0 + peak_width/2) 
    negative_peak_range = (180- peak_width/2,180 +peak_width/2) 

    presim_phase_angles =  epochs_tep.metadata['prestim_phase_deg']
    
    positive_peak_index = np.where(np.logical_or(presim_phase_angles >= positive_peak_range[0], presim_phase_angles <= positive_peak_range[1]))[0]
    negative_peak_index =np.where(np.logical_and(presim_phase_angles >= negative_peak_range[0], presim_phase_angles <= negative_peak_range[1]))[0]

    # Add caterorical variable "peak" to metadata

    epochs_tep.metadata['peak'] = None # Later set to 'neutral'
    epochs_tep.metadata.iloc[positive_peak_index,epochs_tep.metadata.columns.get_loc('peak')] = 'positive'
    epochs_tep.metadata.iloc[negative_peak_index,epochs_tep.metadata.columns.get_loc('peak')] = 'negative'
    epochs_tep.metadata.peak.fillna(value='neutral', inplace=True) # for non-peak angles
    return epochs_tep

# ================= GED-related Functions ======================
import scipy
import copy

def GED_compute_cov(ep_signal,ep_ref,z_threshold, plot = False):
    
    ep_signal.load_data()
    ep_ref.load_data()
    
    
    for ep in [ep_signal,ep_ref]:
        try:
            ep.pick_types(eeg = True)
            units = 'uV'
        except ValueError:
            print ('Make sure all channels are CSD and EMG has been DROPED prior to function')
            units = 'uV/m²'
    

    if np.sign(ep_signal.tmax * ep_signal.tmin) == 1: # select tmin to crop depending if baseline period should be removed
        tmin = None # no baseline timepoints in data
    else:
        tmin = 0 #  baseline timepoints in data
    
    # Get the raw data [N_ep,N_ch,N_points]
    ep_signal_raw = ep_signal.copy().get_data( units=units,tmin = tmin,tmax=None)
    ep_ref_raw = ep_ref.copy().get_data( units= units,tmin = tmin,tmax=None)

    ep_signal_raw = ep_signal_raw[:,:,:]
    ep_ref_raw = ep_ref_raw[:,:,:]

    # Prepare Covariance matrixs [n_epocs,n_ch,n_ch]

    allCovS = np.zeros((ep_signal_raw.shape[0],ep_signal_raw.shape[1],ep_signal_raw.shape[1]))
    allCovR = np.zeros((ep_ref_raw.shape[0],ep_ref_raw.shape[1],ep_ref_raw.shape[1]))

    samples_num = ep_signal_raw.shape[2]

    ## Create covariance matrices for each trial 

    for i,ep in enumerate(ep_signal_raw):
        tmpdat = ep
        # mean-center
        tmpdat = tmpdat-np.mean(tmpdat,axis=1,keepdims=True)
        
        # add to S tensor
        allCovS[i,:,:] = tmpdat@tmpdat.T / samples_num
    

        
    for i,ep in enumerate(ep_ref_raw):
        
        tmpdat = ep
        tmpdat = tmpdat-np.mean(tmpdat,axis=1,keepdims=True)
        allCovR[i,:,:] = tmpdat@tmpdat.T / samples_num

    #Cleaning covariance matrices

    # clean R
    meanR = np.mean(allCovR,axis=0)  # average covariance
    dists = np.zeros(allCovR.shape[0])  # vector of distances to mean
    for segi in range(allCovR.shape[0]):
        r = allCovR[segi,:,:]
        # Euclidean distance
        dists[segi] = np.sqrt( np.sum((r.reshape(1,-1)-meanR.reshape(1,-1))**2) )

    # compute zscored distances
    distsZ = (dists-np.mean(dists)) / np.std(dists)
    print (f'{sum(distsZ>z_threshold)} out of {allCovR.shape[0]} Referance Cov mat has been rejected (Z>{z_threshold})')
    n_droped_R = sum(distsZ>z_threshold)
    
    # finally, average trial-covariances together, excluding outliers
    covR = np.mean( allCovR[distsZ<z_threshold,:,:] ,axis=0)


    ## clean S
    meanS = np.mean(allCovS,axis=0)  # average covariance
    dists = np.zeros(allCovS.shape[0])  # vector of distances to mean
    for segi in range(allCovS.shape[0]):
        r = allCovS[segi,:,:]
        # Euclidean distance
        dists[segi] = np.sqrt( np.sum((r.reshape(1,-1)-meanS.reshape(1,-1))**2) )

    # compute zscored distances
    distsZ = (dists-np.mean(dists)) / np.std(dists)
    print (f'{sum(distsZ>z_threshold)} out of {allCovS.shape[0]} Signal Cov mat has been rejected (Z>{z_threshold})')
    n_droped_S = sum(distsZ>z_threshold)
    # finally, average trial-covariances together, excluding outliers

    covS =  np.mean( allCovS[distsZ<z_threshold,:,:] ,axis=0)
    
    return covS, covR, allCovS,allCovR,n_droped_S,n_droped_R

def topoplotIndie(Values,ep_signal,title='',cbar_label = '',ax=0):
    """_summary_

    Args:
        Values (np.array): componant map for a single compponnt
        ep_signal (mne.Epochs): Contains the channels labels to plot
        title (str, optional): Axis title. Defaults to ''.
        ax (int, optional): Axis to plot in. Defaults to 0.
    """
    from scipy.interpolate import griddata
    from matplotlib import patches

    ## import and convert channel locations from MNE.Epochs instance
    labels = []
    #Th     = []
    #Rd     = []
    x      = []
    y      = []

    #
    for ci in range(len(ep_signal.ch_names)):
        labels.append(ep_signal.info['chs'][ci]['ch_name'])
        
        #Th.append(np.pi/180*chanlocs[0]['theta'][ci][0][0])
        #ep_signal.info['chs'][6]['loc']
        
        ##Rd.append(chanlocs[0]['radius'][ci][0][0])
        x.append( ep_signal.info['chs'][ci]['loc'][0] )
        y.append( ep_signal.info['chs'][ci]['loc'][1] )



    ## remove infinite and NaN values
    # ...
    x = [5*cor for cor in x]
    y = [5*cor for cor in y]

   
    # plotting factors
    headrad = .4 #.4
    plotrad = .5 #.5

    # squeeze coords into head
    squeezefac = headrad/plotrad
    # to plot all inside the head cartoon
    x = np.array(x)*squeezefac
    y = np.array(y)*squeezefac


    ## create grid
    xmin = np.min( [-headrad,np.min(x)] )
    xmax = np.max( [ headrad,np.max(x)] )
    ymin = np.min( [-headrad,np.min(y)] )
    ymax = np.max( [ headrad,np.max(y)] )
    xi   = np.linspace(xmin,xmax,67)
    yi   = np.linspace(ymin,ymax,67)

    # spatially interpolated data
    Xi, Yi = np.mgrid[xmin:xmax:67j,ymin:ymax:67j]
    Zi = griddata(np.array([y,x]).T,Values,(Xi,Yi))
#     f  = interpolate.interp2d(y,x,Values)
#     Zi = f(yi,xi)

    ## Mask out data outside the head
    mask = np.sqrt(Xi**2 + Yi**2) <= headrad
    Zi[mask == 0] = np.nan


    ## create topography
    # make figure
    if ax==0:
        fig  = plt.figure()
        ax   = fig.add_subplot(111, aspect = 1)
    clim = np.max(np.abs(Zi[np.isfinite(Zi)]))*.8
    
    
    the_plot = ax.contourf(yi,xi,Zi,60,cmap=plt.cm.jet,zorder=1, vmin=-clim,vmax=clim)
    cbar = plt.colorbar(the_plot, ax=ax,shrink = 0.5,label = cbar_label)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))

    #plt.colorbar(...., ticks= [] for no ticks)
    # head ring
    circle = patches.Circle(xy=[0,0],radius=headrad,edgecolor='k',facecolor='w',zorder=0)
    ax.add_patch(circle)

    # ears
    circle = patches.Ellipse(xy=[np.min(xi),0],width=.05,height=.2,angle=0,edgecolor='k',facecolor='w',zorder=-1)
    ax.add_patch(circle)
    circle = patches.Ellipse(xy=[np.max(xi),0],width=.05,height=.2,angle=0,edgecolor='k',facecolor='w',zorder=-1)
    ax.add_patch(circle)

    # nose (top, left, right)
    xy = [[0,np.max(yi)+.06], [-.2,.2],[.2,.2]]
    polygon = patches.Polygon(xy=xy,facecolor='w',edgecolor='k',zorder=-1)
    ax.add_patch(polygon)
    
    
    # add the electrode markers
    ax.scatter(x,y,marker='o', c='k', s=15, zorder = 3)

    ax.set_xlim([-.6,.6])
    ax.set_ylim([-.6,.6])
    ax.axis('off')
    ax.set_title(title)
    ax.set_aspect('equal')


def topoplotIndie_improved(Values,ep_signal,title='',cbar_label = '',ax = 0):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """_summary_

    Args:
        Values (np.array): componant map for a single compponnt
        ep_signal (mne.Epochs): Contains the channels labels to plot
        title (str, optional): Axis title. Defaults to ''.
        ax (int, optional): Axis to plot in. Defaults to 0.
    """
    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(Values, ep_signal.info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        axes=ax,
        cmap="coolwarm",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax.images[0]

    # remove the title that would otherwise say "0.000 s"
    ax.set_title(title)
    
    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax)

    # add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar,shrink = 0.5,label = cbar_label)
    #ax_topo.set_xlabel()
    
    plt.tight_layout()

def plot_covs(CovS,CovR,n_droped_S,n_droped_R):
   
    fig,axs = plt.subplots(1,3,figsize=(8,4))
    scaler=1
    # S matrix
    axs[0].imshow(scaler*CovS,cmap='jet')
    axs[0].set_title('S matrix')
    fig.colorbar(axs[0].imshow(scaler*CovS,cmap='jet'),ax=axs[0],location='right', anchor=(0, 0.5), shrink=0.5)

    # R matrix
    axs[1].imshow(scaler*CovR,cmap='jet')
    axs[1].set_title('R matrix')
    fig.colorbar(axs[1].imshow(scaler*CovR,cmap='jet'),ax=axs[1],location='right', anchor=(0, 0.5), shrink=0.5)

    # R^{-1}S
    ratio_mat = np.linalg.inv(scaler*CovR)@(scaler*CovS)
    axs[2].imshow(ratio_mat   ,cmap='jet')
    axs[2].set_title('$R^{-1}S$ matrix')
    fig.colorbar(axs[2].imshow(ratio_mat   ,cmap='jet'),ax=axs[2],location='right', anchor=(0, 0.5), shrink=0.5)
    fig.suptitle(f'{n_droped_S} Signal epochs droped | {n_droped_R} Referance epochs droped')
    plt.tight_layout()
    plt.show()

def GED(covS,covR,epochs,comp_order,sign_flip = False):

    epochs_modify_info = epochs.copy()
    epochs_modify_info.pick_channels(['Fz'])

    #Compute GED
    evals,evecs = scipy.linalg.eigh(covS,covR)
    sidx  = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:,sidx]
    
    # Prepare data for comp time series computation
    try:
        epochs.pick_types(eeg = True)
        units = 'V'
        print('Pick types EEG')
    except ValueError:
        
        units = 'V/m²'

    data = copy.deepcopy(epochs).get_data(units =units)
    
    
    signal_reconstract =  np.zeros((data.shape[0],1,data.shape[2]))
    #WAS signal_reconstract =  np.zeros((data.shape[0],data.shape[1],data.shape[2]))

    for i,epoch in enumerate(data): # Compute comp time series for each epoch
        signal_reconstract[i,:,:] = evecs[:,comp_order].T @ epoch
    

    epochs_modify_info.rename_channels({'Fz': 'GED'})   
    ep = mne.EpochsArray(signal_reconstract,info = epochs_modify_info.info)
    #ep.compute_psd(fmin= 1,fmax=60).plot()

    ### component map
    compmap = evecs[:,comp_order].T @ covS
    if sign_flip: # flip map sign (by the sign of the biggest value)
        se = np.argmax(np.abs( compmap ))
        compmap = compmap * np.sign(compmap[se])
 
    # Lets verify channels names and weigths
    ch_names = epochs.ch_names
    sort_ind =  list(np.argsort(compmap))
    sort_ind.reverse()

    ch_importance =  [ch_names[i] for i in sort_ind]
    compmap_sorted = [compmap[i] for i in sort_ind]

    z = zip(ch_importance, compmap_sorted)
    z = list(z)
    
    df_results = pd.DataFrame(
      {"W": evecs[:,comp_order],
      "Compmap": compmap},
      index = ch_names
)
    df_eigenvectors = pd.DataFrame(data=evecs, index=ch_names, columns=[f'W{i}' for i in range(1, evecs.shape[1] + 1)])
    fig,axs = plt.subplots(2,2,figsize=(10,10))
    axs = axs.flat
    
    
    axs[0].plot(evals,'ks-',markersize=10,markerfacecolor='r')
    axs[0].plot(comp_order,[evals[comp_order]],'ks-',markersize=10,markerfacecolor='g')
    axs[0].set_xlim([0,20])
    axs[0].set_title('GED Components')
    axs[0].set_xlabel('Component number')
    axs[0].set_ylabel('Power ratio ($\lambda$)')
    # Note that the max eigenvalue is <1, 
    # because R has more overall energy than S.

    # GED component
    topoplotIndie_improved(compmap[:, np.newaxis], epochs,title = f'#{comp_order+1} component',cbar_label = '(AU)',ax = axs[2])
    topoplotIndie_improved(evecs[:,comp_order][:, np.newaxis], epochs,title = f'#{comp_order+1} Eigenvector',cbar_label = 'Weights',ax = axs[3])

    # Power spectrta of the componenet
    ep.compute_psd(fmin= 1,fmax=60).plot(axes = axs[1],spatial_colors = False)
    axs[1].set_xticks([5,10,20,40])
    axs[1].set_ylabel('Power (AU)')
    axs[1].set_xlabel('Frequency (Hz)')

    plt.show()
    plt.tight_layout()
    

    return ep, z ,df_results, fig,df_eigenvectors

def transform2D (ep_signal_raw):
    f"""_summary_

    Args:
        ep_signal_raw (np.aaray): [N_ep,N_ch,N_samples]

    Returns:
        np.array: 2D transformed[N,ch,N_samples * N_epochs]
    """
    num_trials, num_electrodes, num_timepoints = ep_signal_raw.shape
    ep_signal_transformed = np.zeros((num_electrodes, num_trials * num_timepoints))

    for i in range(num_electrodes):
        for j in range(num_trials):
            start = j * num_timepoints
            end = start + num_timepoints
            ep_signal_transformed[i, start:end] = ep_signal_raw[j, i, :]
            
    return ep_signal_transformed    

def shrink_reg(covR, gamma):
    """Regularize the Ref matrix pre GED (Mike's paper)
        Because maximizing variance does not necessarily maximize relevance,
        excessive regularization may lead to results that are
        numerically stable but that are
        less sensitive to the desired contrast.
        For this reason, one should use as little regularization as possible 
        but as much as necessary.
    Args:
        covR (2D np.array): Referance matrix for GEP
        gamma (int 0-1): when = 1 Full reg (PCA on S)
    """
    
  
    covRr = covR*(1-gamma) + gamma*np.mean(scipy.linalg.eigh(covR)[0])*np.identity(covR.shape[0])
    
    return covRr

def plot_regularization(CovR):

    fig, axs = plt.subplots(np.arange(0.0, 1.0, 0.2).shape[0],2,constrained_layout=True)

    for row,gamma in enumerate(np.arange(0.0, 1.0, 0.2)):

        covRr = shrink_reg(CovR, gamma)

        #fig.suptitle(f'Regularization for Gamma ={gamma}')
        axs[row,0].set_title('Non-regularization')
        axs[row,0].imshow(CovR)
        fig.colorbar(axs[row,0].imshow(CovR),ax=axs[row,0],location='right', anchor=(0, 0.5), shrink=1)

        axs[row,1].set_title(f'gamma = {np.round(gamma,2)}')
        axs[row,1].imshow(covRr)
        fig.colorbar(axs[row,1].imshow(covRr),ax=axs[row,1],location='right', anchor=(0, 0.5), shrink=1)
        plt.show()
def filterFGx(data,srate,f,fwhm,showplot=False):
    '''
    :: filterFGx   Narrow-band filter via frequency-domain Gaussian
    filtdat,empVals]= filterFGx(data,srate,f,fwhm,showplot=0)


      INPUTS
         data : 1 X time or chans X time
        srate : sampling rate in Hz
            f : peak frequency of filter
         fhwm : standard deviation of filter, 
                defined as full-width at half-maximum in Hz
     showplot : set to true to show the frequency-domain filter shape (default=false)

      OUTPUTS
      filtdat : filtered data
      empVals : the empirical frequency and FWHM (in Hz and in ms)

    Empirical frequency and FWHM depend on the sampling rate and the
     number of time points, and may thus be slightly different from
     the requested values.

     mikexcohen@gmail.com
    '''

    ## compute filter

    # frequencies
    hz = np.linspace(0,srate,data.shape[1])

    # create Gaussian
    s  = fwhm*(2*np.pi-1)/(4*np.pi) # normalized width
    x  = hz-f                       # shifted frequencies
    fx = np.exp(-.5*(x/s)**2)       # gaussian
    fx = fx/np.max(fx)              # gain-normalized

    # apply the filter
    filtdat = np.zeros( np.shape(data) )
    for ci in range(filtdat.shape[0]):
        filtdat[ci,:] = 2*np.real( np.fft.ifft( np.fft.fft(data[ci,:])*fx ) )



    ## compute empirical frequency and standard deviation

    empVals = [0,0,0]

    idx = np.argmin(np.abs(hz-f))
    empVals[0] = hz[idx]

    # find values closest to .5 after MINUS before the peak
    empVals[1] = hz[idx-1+np.argmin(np.abs(fx[idx:]-.5))] - hz[np.argmin(np.abs(fx[:idx]-.5))]

    # also temporal FWHM
    tmp  = np.abs(scipy.signal.hilbert(np.real(np.fft.fftshift(np.fft.ifft(fx)))))
    tmp  = tmp / np.max(tmp)
    tx   = np.arange(0,data.shape[1])/srate
    idxt = np.argmax(tmp)

    empVals[2] = (tx[idxt-1+np.argmin(np.abs(tmp[idxt:]-.5))] - tx[np.argmin(np.abs(tmp[0:idxt]-.5))])*1000



    ## inspect the Gaussian (turned off by default)

    # showplot=True

    if showplot:
        plt.subplot(211)
        plt.plot(hz,fx,'o-')
        xx = [ hz[np.argmin(np.abs(fx[:idx]-.5))], hz[idx-1+np.argmin(np.abs(fx[idx:]-.5))] ]
        yy = [ fx[np.argmin(np.abs(fx[:idx]-.5))], fx[idx-1+np.argmin(np.abs(fx[idx:]-.5))] ]
        plt.plot(xx,yy,'k--')
        plt.xlim([np.max(f-10,0),f+10])

        plt.title('Requested: %g, %g Hz; Empirical: %.2f, %.2f Hz' %(f,fwhm,empVals[0],empVals[1]) )
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude gain')

        plt.subplot(212)
        tmp1 = np.real(np.fft.fftshift(np.fft.ifft(fx)))
        tmp1 = tmp1 / np.max(tmp1)
        tmp2 = np.abs(scipy.signal.hilbert(tmp1))
        plt.plot(tx-np.mean(tx),tmp1, tx-np.mean(tx),tmp2)
        plt.xlim([-empVals[2]*2/1000,empVals[2]*2/1000])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude gain')
    plt.show()
    # 
    
    ## outputs
    return filtdat,empVals

def save_pdf_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()



def create_cmw(srate, frex, fwhm, duration):
    """
    Create a complex Morlet wavelet.

    Parameters:
    srate (int): Sampling rate in Hz.
    frex (float): Frequency of wavelet in Hz.
    fwhm (float): Full width at half maximum of the Gaussian window in seconds.
    duration (float): Total duration of the wavelet in seconds.

    Returns:
    cmw (numpy array): Complex Morlet wavelet.
    time (numpy array): Time vector.
    """
    # Create time vector
    time = np.linspace(-duration / 2, duration / 2, int(duration * srate) + 1)

    # Create a complex-valued sine wave (wavelet)
    sine_wave = np.exp(1j * 2 * np.pi * frex * time)

    # Create Gaussian window
    gaus_win = np.exp((-4 * np.log(2) * time**2) / (fwhm**2))

    # Create Morlet wavelet
    cmw = sine_wave * gaus_win

    # Normalize the wavelet to have unit energy
    cmw = cmw / np.sqrt(np.sum(np.abs(cmw)**2))

    return cmw, time

def plot_wavelet(cmw, time, frex, fwhm):
    """
    Plot the complex Morlet wavelet.

    Parameters:
    cmw (numpy array): Complex Morlet wavelet.
    time (numpy array): Time vector.
    frex (float): Frequency of wavelet in Hz.
    fwhm (float): Full width at half maximum of the Gaussian window in seconds.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, np.real(cmw), 'b', label='Real part')
    plt.plot(time, np.imag(cmw), 'r--', label='Imaginary part')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Complex Morlet wavelet (Frequency: {frex} Hz, FWHM: {fwhm} s)')
    plt.show()



def get_sub_str(sub_num):
    zeros = (3- len(str(sub_num)))*'0' 
    sub_str = f'sub_'+zeros+f'{sub_num}'
    return sub_str



def predict_phase_custom_wavelvet(signal,signal_times,time_to_predict,cmw,frex):


    zero_time_idx = np.where(signal_times == time_to_predict)[0] # look for pulse idx

    zero_time_idx = int(zero_time_idx)

    # Ensure the wavelet length matches the segment length
    wavelet_length = len(cmw)

    half_wavelet_length = len(cmw) // 2

    # Extract the signal segment preceding the zero time point
    if zero_time_idx - wavelet_length < 0:
        raise ValueError("Not enough data points before the zero time point to match the wavelet length.")
    signal = np.squeeze(signal)
    signal_segment = signal[zero_time_idx - wavelet_length: zero_time_idx]

    # Extract the corresponding wavelet segment
    #wavelet_segment = cmw[:len(signal_segment)]

    # Compute the dot product
    dot_product = np.dot(signal_segment, np.conj(cmw))

    # Calculate the phase angle
    phase_angle = np.angle(dot_product)

    print(np.degrees(phase_angle))


    estimate_time =  signal_times[zero_time_idx-half_wavelet_length]
    pulse_time = time_to_predict

    X = estimate_time # initial time point in seconds
    Y = pulse_time  # later time point in seconds
    
    phase_X = phase_angle  # initial phase in radians
    # Compute the phase at time Y
    phase_Y = compute_phase_at_Y(phase_X, frex, X, Y)

    # Convert radians to degrees and make sure to convert negative degrees to 180-360
    phase_X_degrees = np.degrees(phase_X) % 360
    phase_Y_degrees = np.degrees(phase_Y) % 360

    print(f"Phase at time {X}s: {phase_X} radians ({phase_X_degrees} degrees)")
    print(f"Phase at time {Y}s: {phase_Y} radians ({phase_Y_degrees} degrees)")
    result = {
    'estimate_phase': phase_X,
    'estimate_time': estimate_time,
    'prestim_phase_custom': phase_Y

    }
    
    return result

def compare_custom_wavelet_prediction(epochs_tep,phase_epochs_morlet,cmw,epoch_idx,time_to_predict,args):

 

    signal = np.squeeze(epochs_tep.copy().pick(args.source_ch)[epoch_idx]._data)
    phase_mne =  phase_epochs_morlet.copy().pick(args.source_ch)[epoch_idx]._data.squeeze()

    t = epochs_tep.times
    conv_result = np.convolve(signal, cmw, mode='same')

    frex = int(args.freqs[0]) # convert str in a list to int
    points =  predict_phase_custom_wavelvet(signal,epochs_tep.times,time_to_predict,cmw,frex)

    #estimate_phase =  np.deg2rad(points['estimate_phase'])
    estimate_phase =  points['estimate_phase']
    estimate_time = points['estimate_time']
    prestim_phase_deg_custom = points['prestim_phase_custom']

    # Extract the instantaneous phase and amplitude
    instantaneous_phase = np.angle(conv_result)
    instantaneous_amplitude = np.abs(conv_result)

    # Compute the power
    instantaneous_power = instantaneous_amplitude ** 2

    # Plotting the results
    plt.figure(figsize=(12, 8))

    # Original Signal
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal')
    plt.legend()

    # Instantaneous Phase
    plt.subplot(4, 1, 2)
    plt.plot(t, instantaneous_phase, label='Custom Wavelet', color='black')
    plt.plot(t, phase_mne, label='MNE Wavelet', color='orange')
    plt.scatter(estimate_time,estimate_phase, color='black')
    plt.scatter(time_to_predict,prestim_phase_deg_custom, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (radians)')
    plt.title('Instantaneous Phase')
    plt.legend()

    # Instantaneous Amplitude
    plt.subplot(4, 1, 3)
    plt.plot(t, instantaneous_amplitude, label='Instantaneous Amplitude', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Instantaneous Amplitude')
    plt.legend()

    plt.tight_layout()

def compute_phase_at_Y(phase_X, frex, X, Y):
    # Calculate the phase difference
    phase_difference = 2 * np.pi * frex * (Y - X)
    
    # Compute the phase at time Y
    phase_Y = phase_X + phase_difference
    
    # Ensure the phase is within the range [-π, π)
    phase_Y = np.angle(np.exp(1j * phase_Y))
    
    return phase_Y

def plot_average_epochs(epochs_tep, source_ch):
    """
    Plots the average of epochs for each unique peak as subplots.

    Parameters:
    epochs_tep : mne.Epochs
        The epochs data with metadata containing peaks.
    source_ch : list or str
        The channels to pick for averaging.

    Returns:
    fig : matplotlib.figure.Figure
        The figure containing the subplots.
    """
    # Determine the unique peaks
    unique_peaks = epochs_tep.metadata["peak"].unique()

    # Number of unique peaks
    num_peaks = len(unique_peaks)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_peaks, ncols=1, figsize=(10, 3 * num_peaks))

    # Ensure that axes is iterable in case of a single subplot
    if num_peaks == 1:
        axes = [axes]

    # Loop over the unique peaks and plot the average for each
    for ax, peak in zip(axes, unique_peaks):
        avg = epochs_tep[epochs_tep.metadata["peak"] == peak].copy().pick(source_ch).average()
        avg.plot(axes=ax, show=False)
        ax.set_title(f'Average for peak: {peak}')

    plt.tight_layout()
    return fig

def plot_compare_average_epochs(epochs_tep, args, ax):
    # Extract data from MNE epochs
    positive_epochs = epochs_tep[epochs_tep.metadata["peak"] == 'positive'].copy().pick(args.source_ch)
    negative_epochs = epochs_tep[epochs_tep.metadata["peak"] == 'negative'].copy().pick(args.source_ch)

    # Get the number of trials
    num_positive_trials = len(positive_epochs)
    num_negative_trials = len(negative_epochs)

    # Extract the data and compute the average and SEM
    positive_data = positive_epochs.get_data()
    negative_data = negative_epochs.get_data()
    
    positive_avg = positive_data.mean(axis=0).squeeze()
    negative_avg = negative_data.mean(axis=0).squeeze()
    
    positive_sem = positive_data.std(axis=0).squeeze() / np.sqrt(num_positive_trials)
    negative_sem = negative_data.std(axis=0).squeeze() / np.sqrt(num_negative_trials)

    # Extract times from epochs
    times = positive_epochs.times

    # Plot the data
    ax.plot(times, negative_avg, color='blue', label=f'Negative Peaks (n={num_negative_trials})')
    ax.fill_between(times, negative_avg - negative_sem, negative_avg + negative_sem, color='blue', alpha=0.3)
    
    ax.plot(times, positive_avg, color='red', label=f'Positive Peaks (n={num_positive_trials})')
    ax.fill_between(times, positive_avg - positive_sem, positive_avg + positive_sem, color='red', alpha=0.3)

    # Adding a vertical dotted line at time zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Adding titles and labels
    ax.set_title('Average EEG Data for Positive and Negative Peaks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    
    # Adding grid with lines every 100 ms
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))  # 0.1 seconds = 100 ms
    ax.grid(True)


def plot_highlighted_psd_prestim(epochs, psd_kw, welch_kw, highlight_channel):
    """
    Plot the power spectral density (PSD) for all channels, highlighting one specified channel.
    
    Parameters:
    epochs (mne.Epochs): The epochs object containing the data.
    psd_kw (dict): Keyword arguments for the PSD computation.
    welch_kw (dict): Additional keyword arguments for Welch's method.
    highlight_channel (str): The name of the channel to highlight.
    """
    # Compute PSD
    psd = epochs.copy().crop(tmin=-1, tmax=0).compute_psd(**psd_kw, **welch_kw)
    
    # Extract the data and frequencies
    psd_data, freqs = psd.get_data(return_freqs=True)
    channel_names = psd.ch_names
    
    # Convert PSD data to dB (10 * log10 of the data)
    psd_data_db = 10 * np.log10(psd_data)
    
    # Find the index of the channel to highlight
    highlight_idx = channel_names.index(highlight_channel)
    
    # Plot the PSD for all channels
    plt.figure(figsize=(10, 6))
    
    # Plot all channels in grey
    for idx, channel in enumerate(channel_names):
        if idx == highlight_idx:
            plt.plot(freqs, psd_data_db[idx].mean(axis=0), color='r', linewidth=2.5, label=highlight_channel)
        else:
            plt.plot(freqs, psd_data_db[idx].mean(axis=0), color='grey', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Power Spectral Density')
    plt.xscale('log')
    plt.legend()
    plt.show()
