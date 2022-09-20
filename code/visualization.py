import numpy as np
import pickle
import matplotlib.pyplot as plt

from helper_functions import *


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

    


def main():
    original_filepath = "./dataset/data_files/26148_020520_bank0_s1_light.pkl"
    extracted_filepath = "./extracted_data/26148_020520_bank0_s1_light_extracted.pkl"
    
    # pickle_relevant_features(original_filepath)

    with open(extracted_filepath, "rb") as file:
        relevant_features = pickle.load(file)
    
    show_feature_information(relevant_features)



if __name__ == "__main__":
    main()

