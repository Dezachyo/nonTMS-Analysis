#%% Imports
import pandas as pd
import PyQt5
import math
import numpy as np
import seaborn as sns
import pathlib
import mne
from mne.stats import permutation_cluster_test,combine_adjacency,spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency


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


def plot_tfr_conditions(tfr, channel_name, baseline=(-0.4, 0), baseline_mode='percent'):
    """
    Plots the TFRs for each condition in the given channel.

    Parameters:
    - tfr: MNE Time-Frequency Representation (TFR) object
    - channel_name: The name of the channel to plot
    - baseline: Baseline time window (tuple, start and end)
    - baseline_mode: Baseline correction mode (e.g., 'percent', 'zscore')
    """
    # Extract data for the given channel
    channel_index = tfr.ch_names.index(channel_name)

    # Identify unique conditions from the metadata using the correct column name 'event_name'
    conditions = tfr.metadata['event_name'].unique()

    # Initialize vmin and vmax
    vmin = float('inf')
    vmax = float('-inf')

    # First pass to apply baseline correction and determine global vmin and vmax
    for condition in conditions:
        condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
        condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
        tfr_condition = tfr[condition].copy().pick([channel_name])
        tfr_condition.apply_baseline(baseline, mode=baseline_mode)
        data = tfr_condition.data.mean(axis=0)
        vmin = min(vmin, data.min())
        vmax = max(vmax, data.max())

    # Create subplots
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))

    # Plot each TFR in a subplot with the same color scale
    for i, condition in enumerate(conditions):
        condition_indices = tfr.metadata.query(f"event_name == '{condition}'").index
        condition_indices = condition_indices[condition_indices < tfr.data.shape[0]]  # Ensure indices are within bounds
        tfr_condition = tfr[condition].copy().pick([channel_name])
        tfr_condition.apply_baseline(baseline, mode=baseline_mode)
        tfr_condition.average().plot(axes=axes[i], vmin=vmin, vmax=vmax, show=False)
        axes[i].set_title(condition)
        axes[i].set_xlabel('Time (s)')
        if i != 0:
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])  # Remove y-tick labels for all but the first subplot
        else:
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_yticks(tfr_condition.freqs)
            axes[i].set_yticklabels([f'{freq:.1f}' for freq in tfr_condition.freqs])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    return fig

# %%

sub_list = [1,2,3,4,5,6,8,9,10,11]

baseline = (-0.4 , -0.05)
#sub_list = [8]
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

#%%



#%% Permutation test

n_permutations = 1000
p_value_threshold = 0.05 # Cluster correction 

occ_picks = ['PO4','PO8','POz','O2','PO3','PO7','Pz','Oz']

a = [tfr['binding'].data.transpose(1,2,0) for tfr in group_tfr]
b = [tfr['object'].data.transpose(1,2,0) for tfr in group_tfr]

X = [np.stack(a),np.stack(b)]

p = 0.05 # For trial data I used 0.001
df = len(a) - 2
t_threshold = stats.distributions.t.ppf(1 - p / 2, df=df)
# our data at each observation is of shape frequencies × times × channels)

tfr_info =  group_tfr[0]['object'].info
tfr_times = group_tfr[0]['object'].times
tfr_freqs = group_tfr[0]['object'].freqs

adjacency, ch_names = find_ch_adjacency(tfr_info, ch_type= None)

tfr_adjacency = combine_adjacency(len(tfr_freqs), len(tfr_times), adjacency)

# run cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(
    X,
    n_permutations=n_permutations,
    threshold=t_threshold,
    tail=0,
    n_jobs=None,
    buffer_size=None,
    adjacency=tfr_adjacency,
    #stat_fun= ttest_rel_nop
    stat_fun= mne.stats.ttest_ind_no_p
)

T_obs, clusters, p_values, H0 = cluster_stats
good_cluster_inds = np.where(p_values < p_value_threshold)[0]
 #%%
 
 
def calculate_cluster_size(cluster,T_obs, sum_method):
    """
    Calculate the size of a 3D cluster based on the product
    of the number of unique indices along each dimension.

    Parameters:
    - cluster (tuple of 3 np.ndarray): The 3D cluster represented
      by a tuple of arrays containing indices along each dimension.

    Returns:
    - int: The product of unique indices along each dimension.
    """
    
    if sum_method == 'elements':
        result = cluster[0].shape[0] # When using cluster permutation with out_type = 'indicies' (defult), all axis are equal in shape 
    elif sum_method =='cube':
        result = np.prod([len(np.unique(dim)) for dim in cluster])
    elif sum_method == 't':
        result = np.sum(np.abs(T_obs[cluster])) # sum absolute t-values
    
    return result

 
def find_biggest_clusters(clusters,T_obs,sum_method):
    """
    Find the indexes of clusters in descending order based on their sizes.

    Parameters:
    - clusters (list of tuples): List of 3D clusters, each represented
      by a tuple of arrays containing indices along each dimension.

    Returns:
    - list: Indexes of clusters in descending order of sizes.
    """
    # Create a list to store tuples of (index, size) for each cluster
    cluster_sizes = [(i, calculate_cluster_size(cluster,T_obs,sum_method)) for i, cluster in enumerate(clusters)]

    # Sort the list in descending order based on cluster sizes
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)

    # Extract the indexes of the sorted clusters
    sorted_indexes = [index for index, size in sorted_clusters]

    return sorted_indexes,cluster_sizes


def plot_cluster_stats(cluster, tfr):
        
    # unpack cluster information, get unique indices
    freq_inds,time_inds,space_inds  = cluster
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    t_map = T_obs[freq_inds,:,:].mean(axis=0)
    t_map = t_map[time_inds,:].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = tfr.times[time_inds]

    # initialize figure
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    ax_topo = axs

    # create spatial mask
    mask = np.zeros((t_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    t_evoked = mne.EvokedArray(t_map[:, np.newaxis]*10e-4, tfr.info, tmin=0)
    t_evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=ax_topo,
        cmap="coolwarm",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        "Averaged T-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # add new axis for spectrogram
    ax_spec = divider.append_axes("right", size="300%", pad=1.2)
    title = "Cluster #, {0} spectrogram".format(len(ch_inds))
    if len(ch_inds) > 1:
        title += " (mean over channels)"
    T_obs_plot = T_obs[:,:,ch_inds].mean(axis = 2) 
    T_obs_plot_sig = np.zeros(T_obs_plot.shape) * np.nan
    T_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = T_obs_plot[
        tuple(np.meshgrid(freq_inds, time_inds))
    ]

    for f_image, cmap,alpha in zip([T_obs_plot, T_obs_plot_sig], ["coolwarm", "gist_yarg"],[1,0.5]):
        c = ax_spec.imshow(
            f_image,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            alpha = alpha,
            #extent=[tfr.times[0], tfr.times[-1], tfr.freqs[0], tfr.freqs[-1]]
        )
    ax_spec.set_xlabel("Time (ms)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(title)
    ax_spec.axvline(x=np.where(tfr.times == 0)[0][0], color='black', linestyle='--')

    # Fix Freq ticks on the Y-Axis
    freqs = tfr.freqs
    yticks_steps = 5 # Hz
    y_pos = np.array(range(0,len(freqs),yticks_steps))
    y_lab = freqs.astype('int')[range(0,len(freqs),yticks_steps)]
    ax_spec.set_yticks(y_pos,y_lab)

    # Fix Time ticks on the X-Axis

    times = 1e3 * tfr.times  # change unit to ms
    xticks_steps = 200 #ms
    ax_spec.set_xticks(range(0,T_obs.shape[1]-1,xticks_steps) ,range(int(times[0]),int(times[-1]),xticks_steps))
    #ax_spec.vlines(x=500, ymin=0, ymax=len(freqs),colors='green', ls='--', lw=2) 


    # add another colorbar
    ax_colorbar2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel("T-stat")



    ax_stats = divider.append_axes("right", size="300%", pad=1.2)

    ax_stats.hist(T_obs.flatten(), bins=50, color='skyblue', alpha=0.2)  # Plot histogram with 50 bins
    ax_stats.hist(T_obs[cluster], bins=50, color='red')  

    num_data_points_cluster =  T_obs[cluster].shape[0]
    num_data_points_total = T_obs.size

    cluster_ratio = num_data_points_cluster/num_data_points_total * 100
    # Set plot title
    ax_stats.set_title(f'{np.round(cluster_ratio,2)}% of Data Points Selected (red)\n {num_data_points_cluster} Data Points ')  
    ax_stats.set_xlabel('T-stat')  # Set x-axis label
    ax_stats.set_ylabel('')  # Set y-axis label


    # clean up viz
    #mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)
    plt.show()


    plt.show()
    return 0

def plot_H0_and_cluster(H0, clusters, T_obs, p_values, cluster_index, alpha=0.05):
    """
    Plot the H0 distribution and compare a specific cluster against it.

    Parameters:
    -----------
    H0 : ndarray
        The distribution of the maximum cluster-level statistics under the null hypothesis.
    clusters : list
        List of clusters where each cluster is represented by a tuple of arrays of indices.
    T_obs : ndarray
        The observed test statistic array.
    p_values : ndarray
        The p-values for each cluster.
    cluster_index : int
        Index of the cluster to compare.
    alpha : float
        Significance level (default is 0.05).
    
    Returns:
    --------
    None
    """
    if not isinstance(H0, np.ndarray):
        raise ValueError("H0 must be a numpy array")
    if not isinstance(clusters, list):
        raise ValueError("clusters must be a list")
    if not isinstance(T_obs, np.ndarray):
        raise ValueError("T_obs must be a numpy array")
    if not isinstance(p_values, np.ndarray):
        raise ValueError("p_values must be a numpy array")
    if not isinstance(cluster_index, int):
        raise ValueError("cluster_index must be an integer")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")

    # Calculate the threshold for significance
    threshold = np.quantile(H0, 1 - alpha)
    
    # Calculate the cluster statistics
    def _masked_sum(x, c):
        return np.sum(x[c])
    
    cluster_stats = [_masked_sum(T_obs, cluster) for cluster in clusters]
    cluster_stat = cluster_stats[cluster_index]
    cluster_p_value = p_values[cluster_index]

    # Ensure cluster_stat is a single scalar value
    if isinstance(cluster_stat, (list, np.ndarray)):
        cluster_stat = np.sum(cluster_stat)

    # Plot the H0 distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(H0, bins=100, kde=True, color='green', edgecolor='black', alpha=0.5)
    
    # Plot the threshold line
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (p < {alpha})')
    
    # Plot the specific cluster's statistic
    plt.axvline(cluster_stat, color='blue', linestyle='-', linewidth=2, label=f'Cluster {cluster_index} Statistic')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Cluster-level Statistic')
    plt.ylabel('Frequency')
    plt.title(f'H0 Distribution with Cluster Statistic\nCluster {cluster_index} p-value: {cluster_p_value:.3f}')
    
    # Add text box with cluster statistic information
    textstr = f'Cluster {cluster_index} Statistic: {cluster_stat:.2f}'
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()

sorted_indexes,cluster_sizes = find_biggest_clusters(clusters,T_obs,sum_method = 't')

for i in range(0,2):
    cluster_ind = sorted_indexes[i]
    cluster = clusters[cluster_ind]

    plot_cluster_stats(cluster, group_tfr[0]['object'])
    plot_H0_and_cluster(H0, clusters, T_obs, p_values,cluster_ind, alpha=0.05)

        
plt.show()


# %%
