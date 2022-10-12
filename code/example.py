import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from behavioral_clustering import BehavioralClustering
from helper_functions import plot_scaleogram


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
    pickle_example()
    bc = load_example()
    animal_ind = 1
    feature_ind = 2

    power = bc.power[animal_ind][feature_ind]
    power = np.sqrt(power)
    max_ind = 1000
    t = np.arange(len(bc.data[animal_ind])) / bc.capture_framerate
        

    plt.figure()
    plot_scaleogram(power[:,:max_ind], t[:max_ind], bc.freqs, bc.scales)
    plt.show()






if __name__ == "__main__":
    main()


