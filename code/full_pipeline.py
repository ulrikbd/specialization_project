import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
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


def main():
    pickle_pipeline()
    bc = load_pipeline()
    print(bc.embedded.shape)
    print(bc.beh_labels.shape)
    print(bc.fit_pca.shape)
    

if __name__ == "__main__":
    main()


