import numpy as np
import scipy
import math
import vg
import pickle

def pickle_relevant_features(filepath):
    """
    Pickle the relevant features contained in the data
    found in the given filepath

    Parameters:
        filepath (string): path to the original pickled data
    """

    # Get the pickled data for one rat
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    raw_features = extract_relevant_features(data)

    with open("./extracted_data/26148_020520_bank0_s1_light_extracted.pkl", "wb") as file:
        pickle.dump(raw_features, file)


def extract_relevant_features(data):
    """
    Collect the relevant time series which will be used 
    as raw features in the analysis.
    Features extracted:
     - Egocentric head actions relative to body in 3D:
        roll (X), pitch (Y), azimuth (Z)
     - Speed in the XY plane
     - Back angles: pitch (Y), azimuth (Z)
     - Sorted point data??


    Parameters:
        data (dict): All the provided data on one rat

    Returns:
        relevant_features (numpy.ndarray): 
            The relevant data
    """
    
    n_features = 7
    speeds = np.array(data["speeds"][:,2])
    ego3_rotm = np.array(data["ego3_rotm"])

    n_time_points = len(speeds)

    ego3q = np.zeros((n_time_points, 3))
    for i in range(n_time_points):
        ego3q[i,:] = rot2expmap(ego3_rotm[i,:,:])
    
    relevant_features = np.zeros((n_time_points, n_features))
    relevant_features[:,:3] = ego3q
    relevant_features[:,3] = speeds
    relevant_features[:,4:6] = data["back_ang"]
    relevant_features[:,6] = data["sorted_point_data"][:,4,2]
    
    return relevant_features


def rot2expmap(rot_mat):
    """
    Converts rotation matrix to quaternions
    Stolen from github
    """

    expmap = np.zeros(3)
    if np.sum(np.isfinite(rot_mat)) < 9:
        expmap[:] = np.nan
    else:
        d = rot_mat - np.transpose(rot_mat)
        if scipy.linalg.norm(d) > 0.01:
            r0 = np.zeros(3)
            r0[0] = -d[1, 2]
            r0[1] = d[0, 2]
            r0[2] = -d[0, 1]
            sintheta = scipy.linalg.norm(r0) / 2.
            costheta = (np.trace(rot_mat) - 1.) / 2.
            theta = math.atan2(sintheta, costheta)
            r0 = r0 / scipy.linalg.norm(r0)
        else:
            eigval, eigvec = scipy.linalg.eig(rot_mat)
            eigval = np.real(eigval)
            r_idx = np.argmin(np.abs(eigval - 1))
            r0 = np.real(eigvec[:, r_idx])
            theta = vg.angle(r0, np.dot(rot_mat, r0))

        theta = np.fmod(theta + 2*math.pi, 2*math.pi) # Remainder after dividsion (modulo operation)
        if theta > math.pi:
            theta = 2*math.pi - theta
            r0 = -r0
        expmap = r0*theta

    return expmap


