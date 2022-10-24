import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from behavioral_clustering import BehavioralClustering
from helper_functions import get_simulated_data


def pickle_simulated():
    """
    Pickle simulated model to be easily retrieved 
    """
    path = "./models/simulated.pkl"
    bc = get_simulated_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


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
    bc.ds_rate = 1 / bc.capture_framerate
    bc.tsne()
    df["bc"] = bc

    return df



def main():
    # pickle_simulated()
    df = load_simulated()
    embedde = df["bc"].embedded
    plt.scatter(embedde[:,0], embedde[:,1], c = df["labels"])
    plt.show()

if __name__ == "__main__":
    main()
