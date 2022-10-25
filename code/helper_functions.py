import numpy as np
import scipy
import math
import vg
import pickle

from scipy.interpolate import LSQUnivariateSpline

import matplotlib.pyplot as plt
import numpy.random as rd


def spline_regression(y, dim, freq, knot_freq):
    """
    Performs spline regression on a time series y(t).
    The internal knots are centered to get close to 
    equal distance to the endpoints.

    Parameters:
        y (np.ndarray): The time series values
        dim (int): Order of the spline
        freq (float): Frequency of the time points.
        knot_freq (float): Chosen frequency of the
            internal knots.

    Returns:
        (np.ndarray): The regression curve at time 
            points t
    """

    t = np.arange(len(y))
    
    # Calculate the interior knots
    space = int(freq / knot_freq)
    rem = len(t) & space
    knots = np.arange(freq + rem // 2, len(t), space)
    
    spl = LSQUnivariateSpline(t, y, knots, k = dim)
    
    return spl(t)




def pickle_relevant_features(original_filepath, new_filepath):
    """
    Pickle the relevant features contained in the data
    found in the given filepath

    Parameters:
        filepath (string): path to the original pickled data
    """

    # Get the pickled data for one rat
    with open(original_filepath, "rb") as file:
        data = pickle.load(file)
    
    raw_features = extract_relevant_features(data)

    with open(new_filepath, "wb") as file:
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



def plot_scaleogram(power, t, freqs, scales):
    """
    Plots the spectrogram of the power density
    achived using a continuous wavelet transform

    Arguments:
        power (np.ndarray): Estimated power for all 
            time points across multiple scales
    """
     
    # Find appropriate contour levels
    ax = plt.subplot()
    min_level = np.log2(power).min() / 2
    max_level = np.max([np.log2(power).max(), 1])
    level_step = (max_level - min_level) / 9
    levels = np.arange(min_level, max_level + level_step, level_step)

    cnf = plt.contourf(t, np.log2(freqs), np.log2(power),
                 levels = levels, extend="both",
                 cmap=plt.cm.viridis)
    cbar = plt.colorbar(cnf)
    cbar.set_ticks(levels)
    cbar.set_ticklabels(np.char.mod("%.1e", 2.**levels))
    y_ticks = 2**np.arange(np.ceil(np.log2(freqs.min())),
                           np.ceil(np.log2(freqs).max()))
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("Time [" + r'$s$' + "]")
    ax.set_ylabel("Frequency [Hz]")


def generate_example_data():
    """
    Generate example data to test the implementation
    choices. Each distinct behaviour is characterised
    by a set of features, each a superpostion of 
    sine waves corresponding frequencies and amplitudes.
    """

    rd.seed(666)

    # Number of distinct behaviours
    n_b = 10
    # Number of features
    n_f = 5
    # Number of sine-waves per feature
    n_s = 4
    # Length of recorded time series, in seconds
    t_max = 600
    # Capture framerate
    fr = 120
    # Example data
    data = np.zeros(shape = (fr * t_max, n_f))
    # Store labels
    labels = np.zeros(len(data))
    # Time points
    t = np.arange(t_max * fr) / fr
    # Expected length of a behaviour, in seconds
    length = 3
    # Lower frequency bound, Hz
    w_lower = 0.5
    # Upper frequency bound, Hz
    w_upper = 20
    # Mu amplitude parameter
    a_mu = 1 
    # Sigma amplitude parameter
    a_sig = 0.5
    # Determine the corresponding frequencies
    w = w_lower + (w_upper - w_lower) * rd.rand(n_b, n_f, n_s)
    # Determine the corresponding amplitudes
    a = rd.lognormal(mean = a_mu, sigma = a_sig, size = (n_b, n_f, n_s))
    # Gaussian noise added to the features
    noise_std = 0.2

    # Define feature generating function given amplitudes and frequencies
    def feature(freq, ampl, t):
        val = 0
        for i in range(len(freq)):
            val += ampl[i]*np.sin(2*np.pi*freq[i]*t)
        return val + rd.normal(0, noise_std)


    ## Simulate data
    # Choose behavioural changing times
    t_change = np.sort(rd.randint(0, len(data), int(len(data) / length / fr)))
    t_change = np.append(t_change, len(data))
    t_change = np.insert(t_change, 0, 0)
    # Choose behaviour labels
    behaviours = rd.randint(0, n_b, len(t_change))

    # Iterate through behaviours
    for i in range(1, len(t_change)):
        beh = behaviours[i]
        t_c = t_change[i]
        t_vals = t[t_change[i - 1]:t_change[i]]
        labels[t_change[i - 1]:t_change[i]] = np.ones(len(t_vals))*beh

        # Iterate over features
        for j in range(n_f):
            freq = w[beh, j, :]
            ampl = a[beh, j, :]
            temp = lambda time: feature(freq, ampl, time)
            data[t_change[i - 1]:t_change[i], j] = temp(t_vals)
            
    return data, labels, t_change, behaviours, w, t
        

def get_simulated_data():
    """
    Retrieve dictionary of simulated data
    """

    with open("./extracted_data/simulated_data.pkl", "rb") as file:
        df = pickle.load(file)

    return df


def main():
    data, labels, t_change, behaviours, w, t = generate_example_data()
    df = {
            "data": data,
            "labels": labels,
            "t_change": t_change,
            "behaviours": behaviours,
            "freqencies": w,
            "time": t,
    }

    with open("./extracted_data/simulated_data.pkl", "wb") as file:
        pickle.dump(df, file)


if __name__ == "__main__":
    main()
