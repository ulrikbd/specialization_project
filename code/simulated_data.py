import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        get_simulated_data, plot_watershed_heat,
        get_contours,
)

import matplotlib.colors as mcolors
import matplotlib.cm as cm


def pickle_simulated():
    """
    Pickle simulated model to be easily retrieved 
    """
    path = "./models/simulated.pkl"
    bc = get_simulated_pipeline()

    with open(path, "wb") as file: pkl.dump(bc, file)


def load_simulated():
    """
    Loads the pickled simulated model
    """
    path = "./models/simulated.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def get_simulated_pipeline():
    """
    Retrieves the simulated data 
    and performs the analysis, cwt,
    pca, tsne, etc. To be used for 
    analyising the choices in the 
    methodology.
    """

    df = get_simulated_data()
    data = df["data"]
    bc = BehavioralClustering()
    bc.raw_features = [data]
    bc.n_features = data.shape[1]
    bc.remove_nan()
    bc.detrend()
    bc.time_frequency_analysis()
    bc.standardize_features()
    bc.pca()
    bc.ds_rate = 1 / bc.capture_framerate * 20
    bc.tsne()
    df["bc"] = bc

    return df


def plot_simulated_features(df):
    """
    Plotting the simulated features color
    coded by the behaviours.
    """

    t_max = 60*100 + 1
    data = df["data"]
    labels = df["labels"][:t_max]
    t = df["time"][:t_max]
    t_change = df["t_change"][:t_max]
    t_change = np.append(t_change, t_max)
    n_int = np.sum(t_change < t_max)
    behaviours = df["behaviours"][:n_int]
    colors = list(mcolors.TABLEAU_COLORS.values())

    for j in range(data.shape[1]):
        feature = data[:t_max, j]

        plt.figure(figsize = (12, 5))
        ax = plt.subplot(111)
        for i in range(n_int):
            t_low = t_change[i]
            t_high = t_change[i + 1]
            plt.plot(t[t_low:t_high], feature[t_low:t_high],
                     c = colors[behaviours[i]], lw = 0.5)
            plt.xlabel("Time " + r'$[s]$')
        ind = np.unique(behaviours, return_index = True)[1]
        beh_unique = [behaviours[i] + 1 for i in sorted(ind)]
        plt.legend(beh_unique, title = "Behavior",
                   loc = "center left", bbox_to_anchor = (1.01, 0.5))
        leg = ax.get_legend()
        for i in range(len(ind)):
            leg.legendHandles[i].set_color(colors[beh_unique[i] - 1])
        plt.savefig("./figures/simulated/features/color_coded_feature_" + str(j + 1) + ".pdf",
                    bbox_inches = "tight")
        # plt.show()     


def dimensionality_reduction():
    """
    Visualize the difference in two dimensional 
    embedding reached by PCA, MDS, ISOMAP and t-SNE
    on simulated data.
    """

    df = load_simulated()
    bc = df["bc"]
    labels = df["labels"]
    feat = bc.features[0]

    # Downsample
    ind = np.arange(0, len(feat), 120)
    feat = feat[ind,:]
    labels = labels[ind]
    
    ## PCA
    pca = PCA(n_components = 2)
    fit_pca = pca.fit_transform(feat)
    ## MDS
    mds = MDS(n_components = 2)
    fit_mds = mds.fit_transform(feat)
    ## ISOMAP
    isomap = Isomap(n_components = 2, n_neighbors = 20)
    fit_isomap = isomap.fit_transform(feat)
    ## t-SNE
    tsne = TSNE(n_components = 2)
    fit_tsne = tsne.fit_transform(feat)

    plt.figure(figsize = (12, 10))
    ax1 = plt.subplot(221)
    plt.scatter(fit_pca[:,0], fit_pca[:,1], c = labels,
                cmap = "Paired")
    plt.title("PCA")
    ax2 = plt.subplot(222)
    plt.scatter(fit_mds[:,0], fit_mds[:,1], c = labels,
                cmap = "Paired")
    plt.title("MDS")
    ax3 = plt.subplot(223)
    plt.scatter(fit_isomap[:,0], fit_isomap[:,1], c = labels,
                cmap = "Paired")
    plt.title("ISOMAP")
    ax4 = plt.subplot(224)
    plt.scatter(fit_tsne[:,0], fit_tsne[:,1], c = labels,
                cmap = "Paired")
    plt.title("t-SNE")
    plt.savefig("./figures/dimensionality_reduction.pdf", 
                bbox_inches = "tight")
    # plt.show()


def perplexity_tuning():
    """
    Perform the clustering on the simulated data
    varying the perplexity parameter i t-SNE.
    """

    # Perplexity values to be tested
    perp = [1, 5, 30, 50, 200, 500]

    df = load_simulated()
    bc = df["bc"]
    bc.bw = 0.2

    plt.figure(figsize = (12, 8))
    # Iterate over chosen perplexities
    for i in range(len(perp)):
        bc.perp = perp[i]
        bc.tsne()
        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.subplot(2, 3, i + 1)
        plt.title("Perplexity = " + str(perp[i]))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
    plt.savefig("./figures/perplexity_tuning_simulated.pdf",
                bbox_inches = "tight")
    # plt.show()


def test_scaling():
    """
    Try the clustering algorithm with various 
    scaling choices for the wavelet spectrum.
    """

    df = load_simulated()
    bc = df["bc"]
    data = bc.data
    print(data[0].shape)

    
    

def main():
    seaborn.set_theme()
    # pickle_simulated()
    df = load_simulated()

    # plot_simulated_features(df)
    # dimensionality_reduction()
    # perplexity_tuning()
    test_scaling()


if __name__ == "__main__":
    main()
