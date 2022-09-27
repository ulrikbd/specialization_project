import numpy as np
import pickle as pkl

from helper_functions import (
        pickle_relevant_features,
        spline_regression,
)



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
        knot_frequency (float): Interior spacing of the knots
            used in the spline regression
        spline_dim (int): Dimension of the regression spline
            used in detrending the data
        trend (list(np.ndarray)): The trend found in the 
            time series by spline regression
        data_detrended (list(np.ndarray)): The detrended
            time series.
        n_features (int): Number of different recorded
            features, i.e., number of movements recorded
            per animal.
        capture_framerate (int): Frequency (Hz) of the 
            recorded postural time series.
        used_indices (list(np.ndarray)): Indices of non-NaN values
            in the raw features
        data (list(list(np.ndarray))): Starting point for the 
            analysis. Time series data without NaN-values.
    """

    def __init__(self):
        self.train_file_names = []
        self.original_file_path = None
        self.extracted_file_path = None
        self.raw_features = [] 
        self.knot_frequency = 0.5
        self.spline_dim = 3
        self.trend = []
        self.data_detrended = []
        self.n_features = None
        self.capture_framerate = 120 
        self.used_indices = []
        self.data = []
    


    def remove_nan(self):
        """
        Removes each time point where one or more value
        in the time series contain a NaN value.
        Stores the new time series data in 
        an attribute, together with the used row indices.
        """
                
        # Iterate over animals
        for i in range(len(self.raw_features)):
            # Find the indices of usable rows
            self.used_indices.append(~np.isnan(self.raw_features[i]).any(axis =1))
            # Filter the data and add to an attribute
            self.data.append(self.raw_features[i][self.used_indices[i]])



    def detrend(self):
        """
        Detrends the time series individually
        using spline regression, and stores the
        trend and detrended data as attributes
        """
       
       # Iterate over animals
        for i in range(len(self.data)):
            # Perform spline regression on each time series
            trend = [spline_regression(y, self.spline_dim,
                        self.capture_framerate,
                        self.knot_frequency) for y in self.data[i].T]
            self.trend.append(np.array(trend).T)
            self.data_detrended.append(self.data[i] - self.trend[i])

        
    
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
        Stores number of features as attribute.

        Attrubutes:
            train_files (List(str)): Names of training files
        """

        self.train_file_names = train_files
        for filename in train_files:
            with open(self.extracted_file_path + 
                      filename.split(".")[0] + "_extracted.pkl",
                      "rb") as file:
                self.raw_features.append(pkl.load(file))

        # Store the number of features
        self.n_features = self.raw_features[0].shape[1]


     

    
