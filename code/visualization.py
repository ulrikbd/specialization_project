import numpy as np
import pickle
import matplotlib.pyplot as plt

from helper_functions import *



def main():
    original_filepath = "./dataset/data_files/26148_020520_bank0_s1_light.pkl"
    extracted_filepath = "./extracted_data/26148_020520_bank0_s1_light_extracted.pkl"
    
    # pickle_relevant_features(original_filepath)

    with open(extracted_filepath, "rb") as file:
        relevant_features = pickle.load(file)
    
    print(relevant_features.shape)



if __name__ == "__main__":
    main()

