import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
        scale_power_spectrum, estimate_pdf,
        get_contours, get_watershed_labels,
        plot_watershed_heat,
)

from scipy.spatial.distance import cdist


def get_example_pipeline():
    filenames = [
        "26148_020520_bank0_s1_light.pkl",
        "26148_030520_bank0_s2_light.pkl",
    ]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")
    
    bc.load_relevant_features(filenames)
    bc.remove_nan()
    bc.detrend()
    bc.time_frequency_analysis()
    bc.pca()
    bc.tsne()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()
    return bc


def pickle_example():
    """
    Pickle example model to be easily retrieved 
    """
    path = "./models/example.pkl"
    bc = get_example_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_example():
    """
    Loads the pickled example model
    """
    path = "./models/example.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc

def test_scaling():
    """
    Try the clustering algorithm with various 
    scaling choices for the wavelet spectrum.
    1. No square root + standardization
    2. Square root + standardization
    3. No square root + no standardization
    4. Square root + no standardization
    """

    ## 1 
    df = load_simulated()
    bc = df["bc"]
    labels = df["labels"]
    ind = np.arange(0, len(bc.fit_pca),
                    int(bc.capture_framerate * bc.ds_rate))
    labels = labels[ind]
    emb_1 = scale_power_spectrum(
            bc, sqrt = False, standardize = True)
    ## 2 
    df = load_simulated()
    bc = df["bc"]
    emb_2 = scale_power_spectrum(
            bc, sqrt = True, standardize = True)
    ## 3 
    df = load_simulated()
    bc = df["bc"]
    emb_3 = scale_power_spectrum(
            bc, sqrt = False, standardize = False)
    ## 4 
    df = load_simulated()
    bc = df["bc"]
    emb_4 = scale_power_spectrum(
            bc, sqrt = True, standardize = False)

    plt.figure(figsize = (12, 10))
    plt.subplot(221)
    plt.scatter(emb_1[:,0], emb_1[:,1], c = labels,
                cmap = "Paired")
    plt.title("No square root w/standarization")
    plt.subplot(222)
    plt.scatter(emb_2[:,0], emb_2[:,1], c = labels,
                cmap = "Paired")
    plt.title("Square root w/standarization")
    plt.subplot(223)
    plt.scatter(emb_3[:,0], emb_3[:,1], c = labels,
                cmap = "Paired")
    plt.title("No square root wo/standarization")
    plt.subplot(224)
    plt.scatter(emb_4[:,0], emb_4[:,1], c = labels,
                cmap = "Paired")
    plt.title("Square root wo/standarization")
    plt.savefig("./figures/test_scaling.pdf", 
                bbox_inches = "tight")
    plt.show()


def test_scaling_example():
    """
    Try the clustering algorithm with various 
    scaling choices for the wavelet spectrum.
    1. Square root + standardization
    2. Square root + no standardization
    """

    ## 1 
    bc = load_example()
    emb_1 = scale_power_spectrum(
            bc, sqrt = True, standardize = True)
    ## 2 
    bc = load_example()
    emb_2 = scale_power_spectrum(
            bc, sqrt = True, standardize = False)


    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plt.scatter(emb_1[:,0], emb_1[:,1], s = 2)
    plt.title("Square root w/standarization")
    plt.subplot(122)
    plt.scatter(emb_2[:,0], emb_2[:,1], s = 2)
    plt.title("Square root wo/standarization")
    plt.savefig("./figures/test_scaling_example.pdf", 
                bbox_inches = "tight")
    plt.show()


def test_pre_embedding():
    """
    Test effects of pre-embedding all points 
    into t-SNE plane by euclidean distance in PCA
    space.
    """

    bc = load_example()

    # Principal component scores for 
    # points used to find t-SNE embedding.
    pca_train = bc.fit_pca[bc.tsne_ind,:]

    # Create storage for embeddings
    emb_total = np.zeros(shape = (len(bc.fit_pca), 2))

    # Iterate over all time points
    for i in range(len(bc.fit_pca)):
        # Find closest time point in PCA space
        dist = cdist(bc.fit_pca[i,:][np.newaxis,:],
                     pca_train)

        # Choose embedding corresponding to this
        # time point
        emb_total[i] = bc.embedded[dist.argmin()]

    
    # Apply kernel density estimation on all points
    kde, grid = estimate_pdf(emb_total, bc.bw, bc.border,
                             500j)

    ws_labels = get_watershed_labels(kde)
    contours = get_contours(kde, ws_labels)
    bc_contours = get_contours(bc.kde, bc.ws_labels)
        
    plt.figure(figsize = (12, 10))
    plt.subplot(221)
    plt.scatter(bc.embedded[:,0], bc.embedded[:,1], s = 2)
    plt.title("Training data, " + str(len(bc.tsne_ind)) + " points")
    plt.subplot(222)
    plt.scatter(emb_total[:,0], emb_total[:,1], s = 2)
    plt.title("Embedded data, " + str(len(bc.fit_pca)) + " points")
    plt.subplot(223)
    plot_watershed_heat(bc.embedded, bc.kde,
                        bc_contours, bc.border)
    plt.title("Segmentation wo/pre embedding") 
    plt.subplot(224)
    plot_watershed_heat(emb_total, kde,
                        contours, bc.border)
    plt.title("Segmentation w/pre embedding") 
    plt.savefig("./figures/test_pre_embed.pdf",bbox_inches = "tight")
    plt.show()


def main():
    seaborn.set_theme()
    # pickle_example()
    bc = load_example()
    # test_scaling_example()
    # test_pre_embedding()

    

if __name__ == "__main__":
    main()


