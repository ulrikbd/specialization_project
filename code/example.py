import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
)


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
    bc.kernel_density_estimation(100j)
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


def main():
    # pickle_example()
    bc = load_example()
    

if __name__ == "__main__":
    main()


