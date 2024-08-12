#%% imports
import pandas as pd
import PyQt5
import math
import numpy as np
import pathlib
import mne
from mne.stats import permutation_cluster_test,combine_adjacency,spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
import argparse

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind
import mne.stats
import scipy.stats as stats
from tqdm import tqdm
from subfunctions import get_sub_str, cprint
from subfunctions import create_phase_epochs_morlet,plot_peak_variation,plot_prestim_phase_dist_to_ax
from subfunctions import add_prestim_col, add_peak_col
from subfunctions import create_cmw, plot_wavelet

# import custom wavelet phase prediction functions
from subfunctions import compare_custom_wavelet_prediction, compute_phase_at_Y, predict_phase_custom_wavelvet, plot_average_epochs, plot_compare_average_epochs,plot_highlighted_psd_prestim, point_by_conv

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
#%%


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description')

# Add arguments
parser.add_argument('--duration', type=int, default=1)
parser.add_argument('--source_ch', type=list, default=['Afz'])
parser.add_argument('--freqs', type=list, default=['6'])
parser.add_argument('--n_cycles', type=int, default=2)
parser.add_argument('--prestim_time', type=int, default=5) # for phase estimation (ms)
parser.add_argument('--peak_width', type=int, default=80,help='±peak_width/2 to both direction') #±peak_width/2 to both direction
parser.add_argument('--csd_transform', type=bool, default=True) 
parser.add_argument('--auto_reject_rest', type=bool, default=True) 

# Parse the command-line arguments
args = parser.parse_args('')


#sub_list = [1,2,3,4,5,6,8,9,10,11]
sub_list = [5,6,8]
# Define file paths and subject list
current_path = pathlib.Path().absolute()
save_path = current_path / 'prepro'

tep_files = [save_path /get_sub_str(sub_num) / f'{get_sub_str(sub_num)}_prepro-epo.fif' for sub_num in sub_list]


# Read the TFR data
group_data = []
for fname,sub_num in zip(tep_files,sub_list):
    
    epochs_tep = mne.read_epochs(fname)
    if args.csd_transform:
        epochs_tep = mne.preprocessing.compute_current_source_density(epochs_tep)
    
    
    # Select only binding trial
    epochs_tep = epochs_tep['binding']
    epochs_tep.metadata.reset_index(drop=True, inplace=True)
    # Create custom wavelet 
    srate = 1000  # in Hz
    frex = int(args.freqs[0])  # frequency of wavelet, in Hz
    fwhm = 0.3  # width of the Gaussian in seconds
    duration = 0.4  # total duration of the wavelet in seconds

    time_to_predict =  0.6

    cmw,cmw_time = create_cmw(srate, frex, fwhm, duration)
    plot_wavelet(cmw, cmw_time, frex, fwhm)

    # TODO compute peak activity
    # Parameters for PSD calclutation 
    time_window = 3
    n_per_seg = time_window*epochs_tep.info['sfreq']
    n_fft = n_per_seg*2 # Double the time point for zero pedding


    welch_kw = {'n_fft' :int(n_fft)
            ,'n_per_seg': int(n_per_seg)
                }
    psd_kw = {'method' :'welch'
            ,'fmin': 0.5
            , 'fmax': 60
                }
        
    psd = epochs_tep.copy().crop(tmin =-1 ,tmax = 0).compute_psd(**psd_kw,**welch_kw)

    fig, axs = plt.subplots(nrows=2)
    psd.plot(xscale = 'log',axes = axs[0])
    psd.plot(xscale = 'log',picks = args.source_ch,axes = axs[1])

    plot_highlighted_psd_prestim(epochs_tep, psd_kw, welch_kw, args.source_ch[0])

    phase_epochs_morlet,_ = create_phase_epochs_morlet(epochs_tep,args.freqs,args.n_cycles)# Phase using Morlet Wavelet

    epochs_tep = add_prestim_col(epochs_tep,phase_epochs_morlet,time_to_predict,args.source_ch)



    # calulate and predict phase with custom wavelet
    custom_phase = []
    custom_phase_conv = []

    # Iterate over each epoch
    for epoch_data in epochs_tep.copy().pick('Afz'):
        result = predict_phase_custom_wavelvet(epoch_data,epochs_tep.times,time_to_predict,cmw,frex)
        result_conv = point_by_conv(epoch_data,epochs_tep.times,time_to_predict,cmw,frex)
        custom_phase.append(result)
        custom_phase_conv.append(result_conv)

    # Create a DataFrame from the list of dictionaries
    df_custom_phase = pd.DataFrame(custom_phase)
    df_custom_phase_conv = pd.DataFrame(custom_phase_conv)


    epochs_tep.metadata['prestim_phase_custom'] = df_custom_phase['prestim_phase_custom']
    epochs_tep.metadata['prestim_phase_deg_custom'] = np.degrees(df_custom_phase['prestim_phase_custom']) %360
    #epochs_tep.metadata['prestim_phase_deg_custom_conv'] = df_custom_phase_conv['point_phase_conv_deg']

    epochs_tep.metadata = pd.concat([epochs_tep.metadata,df_custom_phase_conv],axis=1)
    epochs_tep.metadata.reset_index(drop=True, inplace=True)


    #epochs_tep = add_peak_col(epochs_tep,args.peak_width,'prestim_phase_deg')
    epochs_tep = add_peak_col(epochs_tep,args.peak_width,'point_phase_conv_deg')

    # Function to calculate the circular difference between two angles
    def circular_difference(angle1, angle2):
        diff = angle1 - angle2
        return (diff + 180) % 360 - 180

    # Calculate the circular difference between the two angle columns
    epochs_tep.metadata['angle_diff_deg'] = epochs_tep.metadata.apply(
        lambda row: circular_difference(row['point_phase_conv_deg'], row['prestim_phase_deg_custom']),
        axis=1
    )

    group_data.append(epochs_tep)
        
    epoch_idx = 0
    compare_custom_wavelet_prediction(epochs_tep,phase_epochs_morlet,cmw,epoch_idx,time_to_predict,args)


    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 10))

    # Create a polar subplot
    ax1 = fig.add_subplot(211, polar=True)  # 1 row, 2 columns, 1st subplot (polar)

    # Create a regular Cartesian subplot
    ax2 = fig.add_subplot(212)  # 1 row, 2 columns, 2nd subplot (regular)
    
    plot_phase_amp_vectors_with_mean_and_angle_sd(epochs_tep.metadata, 'angle_diff_deg', 'point_amp_conv' ,ax1)
    plot_compare_average_epochs(epochs_tep.copy().crop(time_to_predict - 0.5 , time_to_predict+0.2),
                                args,
                                time_to_predict,
                                ax2)

# %%






def calculate_means(df, angle_column, amp_column):
    """
    Calculate both the arithmetic mean of the amplitudes and the mean vector considering phase and amplitude.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the angle and amplitude data.
    angle_column (str): The name of the column containing angle differences in degrees.
    amp_column (str): The name of the column containing amplitude values.
    
    Returns:
    tuple: (mean_amplitude, mean_angle, mean_vector_amplitude, std_angle_diff)
    - mean_amplitude: Arithmetic mean of the amplitude values.
    - mean_angle: The angle of the resultant mean vector in radians.
    - mean_vector_amplitude: The magnitude of the resultant mean vector.
    - std_angle_diff: The standard deviation of the angle differences in radians.
    """
    
    # Convert the angle differences to radians
    angle_diff_rad = np.deg2rad(df[angle_column])

    # Get the amplitude values
    amp_values = df[amp_column]

    # Compute the arithmetic mean of the amplitude values
    mean_amplitude = np.mean(amp_values)

    # Compute the components of the vectors for averaging
    x_components = amp_values * np.cos(angle_diff_rad)
    y_components = amp_values * np.sin(angle_diff_rad)

    # Compute the mean vector
    mean_x = np.mean(x_components)
    mean_y = np.mean(y_components)
    mean_vector_amplitude = np.sqrt(mean_x**2 + mean_y**2)
    mean_angle = np.arctan2(mean_y, mean_x)

    # Compute the standard deviation of the angle differences
    std_angle_diff = np.std(angle_diff_rad)

    return mean_amplitude, mean_angle, mean_vector_amplitude, std_angle_diff






def plot_phase_amp_vectors_with_mean_and_angle_sd(df, angle_column, amp_column, ax):
    """
    Plots phase-amplitude combinations as vectors on a polar plot using the provided ax and adds the mean vector 
    with the standard deviation (SD) of the angle differences as a shaded area.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the angle and amplitude data.
    angle_column (str): The name of the column containing angle differences in degrees.
    amp_column (str): The name of the column containing amplitude values.
    ax (matplotlib.axes._subplots.PolarAxesSubplot): The axis to plot on.
    """
    
    # Calculate means using the provided function
    mean_amplitude, mean_angle, mean_vector_amplitude, std_angle_diff = calculate_means(df, angle_column, amp_column)
    
    # Convert the angle differences to radians
    angle_diff_rad = np.deg2rad(df[angle_column])

    # Get the amplitude values
    amp_values = df[amp_column]

    # Plot each vector using the plot method with the chosen darker blue color
    for angle, amplitude in zip(angle_diff_rad, amp_values):
        ax.plot([angle, angle], [0, amplitude], color='#4682B4')  # Darker blue

    # Plot the mean vector in red
    ax.plot([mean_angle, mean_angle], [0, mean_vector_amplitude], color='red', linewidth=2, label='Mean Vector')

    # Plot the SD as a shaded area around the mean angle
    theta = np.linspace(mean_angle - std_angle_diff, mean_angle + std_angle_diff, 100)
    r = np.full_like(theta, mean_vector_amplitude)
    #ax.fill_between(theta, 0, r, color='red', alpha=0.3, label='Angle Difference ± 1 SD')

    # Set theta zero location to 'N' to ensure 0 degrees is at the top (North)
    ax.set_theta_zero_location('N')

    # Ensure angles increase clockwise (which is typical for polar plots)
    ax.set_theta_direction(-1)

    # Set the radial limit to better visualize the vectors
    ax.set_ylim(0, np.max(amp_values) * 1.2)

    # Add labels, legend, and title
    ax.set_title('Phase-Amplitude Vectors with Mean and Angle SD')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

subject_dfs = [epochs_tep.metadata, epochs_tep.metadata]

nrows = 1
ncols = 2


# Create the figure and subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'polar': True}, figsize=(15, 10))
axes = axes.flatten()  # Flatten the array of axes for easy iteration

# Iterate over each subject DataFrame and corresponding subplot axis
for i, (df, ax) in enumerate(zip(subject_dfs, axes)):
    plot_phase_amp_vectors_with_mean_and_angle_sd(df, 'angle_diff_deg', 'point_amp_conv' ,ax)
    ax.set_title(f'Subject {i+1}')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#%% 


def polar_scatter_plot_with_amp(df, angle_column, amp_column, ax, angle_bins=36, r_min=1.5, r_max=3.5):
    """
    Creates a polar scatter plot with color representing amplitude values on the provided ax.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the angle and amplitude data.
    angle_column (str): The name of the column containing angle differences in degrees.
    amp_column (str): The name of the column containing amplitude values.
    ax (matplotlib.axes._subplots.PolarAxesSubplot): The axis to plot on.
    angle_bins (int): Number of bins for the angle histogram. Default is 36.
    r_min (float): Minimum radial distance for scatter points. Default is 1.5.
    r_max (float): Maximum radial distance for scatter points. Default is 3.5.
    """
    
    # Convert the angle differences to radians
    angle_diff_rad = np.deg2rad(df[angle_column])

    # Normalize the amplitude values to the range [0, 1]
    amp_values = df[amp_column]
    norm_amp_values = (amp_values - amp_values.min()) / (amp_values.max() - amp_values.min())

    # Use the 'Blues' colormap
    cmap = cm.get_cmap('Blues')

    # Convert normalized amplitude values to colors using the colormap
    colors = cmap(norm_amp_values)

    # Plot the distribution of angle differences as a histogram
    n, bins, patches = ax.hist(angle_diff_rad, bins=angle_bins, edgecolor='black', alpha=0.3)

    # Adjust the radial positions for the scatter plot to spread them out
    r_values = np.interp(angle_diff_rad, (angle_diff_rad.min(), angle_diff_rad.max()), (r_min, r_max))

    # Add a scatter plot to show individual observations
    sc = ax.scatter(angle_diff_rad, r_values, c=colors, s=50, cmap=cmap, alpha=0.75, edgecolor='black')

    # Add labels and title
    ax.set_theta_zero_location('N')  # Zero degrees at the top (North)
    ax.set_theta_direction(-1)  # Angles increase clockwise
    ax.set_title('Angle Differences with Amplitude Values')

    # Optionally return the scatter object for further customization
    return sc

# Create a figure with one subplot
fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True}, figsize=(8, 8))

# Plot on the ax using the polar_scatter_plot_with_amp function
polar_scatter_plot_with_amp(epochs_tep.metadata, 'angle_diff_deg', 'point_amp_conv', ax)

# Show the plot
plt.show()