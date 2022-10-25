import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

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
    bc.plot()
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

    t_max = 60*90
    feature = df["data"][:t_max,3]
    labels = df["labels"][:t_max]
    t = df["time"][:t_max]
    t_change = df["t_change"][:t_max]
    t_change = np.append(t_change, t_max)
    n_int = np.sum(t_change < t_max)
    behaviours = df["behaviours"][:n_int]
    colors = list(mcolors.TABLEAU_COLORS.values())

    plt.figure(figsize = (12, 5))
    for i in range(n_int):
        t_low = t_change[i]
        t_high = t_change[i + 1]
        plt.plot(t[t_low:t_high], feature[t_low:t_high],
                 c = colors[behaviours[i]])
    ind = np.unique(behaviours, return_index = True)[1]
    plt.legend([behaviours[i] for i in ind])
    plt.show()     
    
        

    

    


def main():
    # pickle_simulated()
    df = load_simulated()

    plot_simulated_features(df)


if __name__ == "__main__":
    main()
