import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn

from helper_functions import *

from behavioral_clustering import BehavioralClustering

from example import get_example_pipeline

def describe(array):
    """
    Show descriptive statistics for an numpy.ndarray
    """
    n_nan = np.sum(np.isnan(array))
    print(f'Length: {len(array)}, #NaN: {n_nan}, %NaN: {n_nan / len(array) * 100:.2f}%')
    print(f'Min: {np.nanmin(array):.2f}, Max: {np.nanmax(array):.2f}, '
          f'Mean: {np.nanmean(array):.2f}, Median: {np.nanmedian(array):.2f}')


def show_feature_information(relevant_features):
    """
    Print out useful information about the features
    """
    for i in range(relevant_features.shape[1]):
        print(f'Feature {i + 1}:')
        describe(relevant_features[:,i])
        print()

    
def get_extracted_filename(original_name):
    """
    Returns filename which specifies the relevant features
    are extracted.
    """
    return original_name.split(".")[0] + "_extracted.pkl"



def plot_detrending():
    """
    Visualizes the detrending performed on the 
    actual data.
    """
    
    bc = get_example_pipeline()

    animal_ind = 1
    feat_ind = 1
    y = bc.data[animal_ind][:, feat_ind]
    x = bc.data_detrended[animal_ind][:, feat_ind]
    s = bc.trend[animal_ind][:, feat_ind]
    t = np.arange(len(y)) / bc.capture_framerate

    plt.figure(figsize = (12, 10))
    ax1 = plt.subplot(2, 1, 1)
    plt.xticks(np.arange(0, t[-1], 60))
    plt.plot(t, y, c = "k", lw = 0.7,  label = r'$y(t)$')
    plt.plot(t, s, c = "r", label = r'$s_3(t)$')
    plt.tick_params('x', labelbottom = False)
    plt.legend(loc = "lower right", prop = {"size": 15},
               frameon = False, framealpha = 0)

    ax2 = plt.subplot(2, 1, 2, sharex = ax1)
    plt.plot(t, x, c = "b", lw = 0.6, label = r'$x(t)$')
    plt.xlabel(r'$t$' + " [s]")
    plt.subplots_adjust(hspace = 0.03)
    plt.legend(loc = "upper right", prop = {"size": 15},
               frameon = False, framealpha = 0)
    plt.savefig("./figures/detrending_bc.pdf", bbox_inches = "tight")
    # plt.show()



def main():
    seaborn.set_theme()
    plot_detrending()



if __name__ == "__main__":
    main()

