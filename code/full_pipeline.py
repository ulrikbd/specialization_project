import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
        get_contours, plot_watershed_heat,
)


def get_pipeline():
    cwd = os.getcwd()
    filenames = [path.split("/")[-1] for path in
                 os.listdir(cwd + "/dataset/data_files/")]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")

    bc.load_relevant_features(filenames)
    # 10 first animals, else RAM overload
    bc.raw_features = bc.raw_features[:10]
    bc.remove_nan()
    print("Detrend")
    bc.detrend()
    print("Time frequency analysis")
    bc.time_frequency_analysis()
    print("Pca")
    bc.pca()
    bc.ds_rate = 1
    print("tsne")
    bc.tsne()
    print("kde")
    bc.kernel_density_estimation(100j)
    print("watershed")
    bc.watershed_segmentation()
    bc.classify()
    return bc


def pickle_pipeline():
    """
    Pickle full model to be easily retrieved 
    """
    path = "./models/full_pipeline.pkl"
    bc = get_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_pipeline():
    """
    Loads the pickled full model
    """
    path = "./models/full_pipeline.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def perplexity_tuning_full():
    """
    Perform the clustering on the full pipeline
    varying the perplexity parameter i t-SNE.
    """

    # Perplexity values to be tested
    perp = [1, 5, 30, 50, 200, 500]

    bc = load_pipeline()
    bc.bw = 0.01

    plt.figure(figsize = (12, 8))
    # Iterate over chosen perplexities
    for i in range(len(perp)):
        bc.perp = perp[i]
        print("Perplexity:", perp[i])
        bc.tsne()
        bc.kernel_density_estimation(100j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.subplot(2, 3, i + 1)
        plt.title("Perplexity = " + str(perp[i]))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
    plt.savefig("./figures/perplexity_tuning_full_pipeline.pdf", 
                bbox_inches = "tight")


def bandwidth_tuning():
    """
    Plot the resulting clusters for several values of
    the bandwidth parameter used in the kernel 
    density estimation.
    """

    bc = load_pipeline()

    bandwidths = np.logspace(-5, -1, 10)

    for bandwidth in bandwidths:
        bc.bw = bandwidth

        bc.tsne()
        bc.kernel_density_estimation(100j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.figure()
        plt.title("Bandwidth = " + str(bandwidth))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
        plt.savefig("./figures/bandwidth_tuning/clustering_bw_" + str(bc.bw) + ".pdf", 
                bbox_inches = "tight")


def main():
    sns.set_theme()
    # pickle_pipeline()
    bc = load_pipeline()

    # perplexity_tuning_full()
    bandwidth_tuning()
    

if __name__ == "__main__":
    main()


