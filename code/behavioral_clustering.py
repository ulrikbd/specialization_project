import numpy as np
import pickle as pkl

from helper_functions import pickle_relevant_features



class BehavioralClustering():
    """
    Wrapper class for clustering distinct behaviours
    in rats from recorded postural dynamics.

    Attributes:
        train_file_names (List(str)): Name of raw data 
            to be used in the clustering
        original_file_path (str): Location of the training
            files.
        extracted_file_path (str): Location of the pickled 
            files containing only relevant features.
        raw_features (List(np.ndarray)): The features relevant to 
            the analysis, straight from the input files.
    """

    def __init__(self):
        self.train_file_names = []
        self.original_file_path = None
        self.extracted_file_path = None
        self.raw_features = [] 
    
    
    def set_original_file_path(self, original_file_path):
        self.original_file_path = original_file_path


    def set_extracted_file_path(self, extracted_file_path):
        self.extracted_file_path = extracted_file_path


    def pickle_relevant_features(self, file_names):
        """
        Extract the relevant raw features from the original
        data, pickle it, and dump it in the location
        specified in the extraced_file_path attribute

        Attributes:
            file_names (List(str)): File names of original data
        """
        for name in file_names:
            pickle_relevant_features(self.original_file_path + name,
                                     self.extracted_file_path +
                                     name.split(".")[0] + 
                                     "_extracted.pkl")


    def load_relevant_features(self, train_files):
        """
        Loads the pickled raw relevant features into the 
        instance attribute to be used for training.

        Attrubutes:
            train_files (List(str)): Names of training files
        """

        self.train_file_names = train_files
        for filename in train_files:
            with open(self.extracted_file_path + 
                      filename.split(".")[0] + "_extracted.pkl",
                      "rb") as file:
                self.raw_features.append(pkl.load(file))




     

    
