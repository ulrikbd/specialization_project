import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn

from behavioral_clustering import BehavioralClustering
from helper_functions import get_simulated_data

import matplotlib.colors as mcolors


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


    
        

    

    


def main():
    seaborn.set_theme()
    pickle_simulated()
    df = load_simulated()

    # plot_simulated_features(df)


if __name__ == "__main__":
    main()
