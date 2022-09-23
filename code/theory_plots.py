import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import periodogram



def underlying(t):
    """ Underlying time series """
    return np.sin(2*np.pi*t) + 1/4*np.cos(6*np.pi*t) + t / 3


def plot_non_stationary_time_series(t, y):
    """ Example plot of a non-stationary time series """ 

    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", label = "Observed series")
    plt.xlabel(r'$t$')
    plt.savefig("./figures/time_series_example.pdf", bbox_inches = "tight")
    # plt.show()


def plot_linear_trend(t, y):
    """ 
    Plot the linear trend in non-stationary time series
    found by linear regression 
    """

    reg = LinearRegression().fit(t, y) 

    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", label = "Observed series")
    # plt.plot(t, underlying(t), c = "r", label = "Underlying function")
    plt.plot(t, reg.predict(t), c = "b", label = "Linear trend")
    plt.xlabel(r'$t$')
    plt.savefig("./figures/time_series_example_with_trend.pdf", bbox_inches = "tight")
    # plt.show()


def plot_cubic_splines(t, y):
    """ 
    Apply cubic spline regression using 
    least squares on the data
    """

    deg = 3
    t = np.reshape(t, t.size)  # Need to flatten the array
    n, t0, tn = len(t), t[0], t[-1]
    m1, m2 = 10, 50
    knots10 = np.linspace(t0, tn, n // m1)[1:-1,]
    knots50 = np.linspace(t0, tn, n // m2)[1:-1,]

    spl10 = LSQUnivariateSpline(t, y, knots10, k = deg)
    spl50 = LSQUnivariateSpline(t, y, knots50, k = deg)
    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", alpha = 0.8,
             label = "Observed series")
    plt.plot(t, spl10(t), c = "r", lw = 3,
             label = "Cubic spline: " + r'$n =$' + str(n) + r'$, m =$' + str(n // m1))
    plt.plot(t, spl50(t), c = "g", lw = 3,
             label = "Cubic spline: " + r'$n =$' + str(n) + r'$, m =$' + str(n // m2))
    plt.xlabel(r'$t$')
    plt.legend()
    plt.savefig("./figures/cubic_splines.pdf", bbox_inches="tight")
    # plt.show()


def plot_periodogram(t, y):
    """
    Plot the periodogram, i.e., the estimated power spectral density 
    """
    fs = 1/(t[1] - t[0])
    f, Pxx_den = periodogram(y ,fs, window = "boxcar",
                             detrend = "linear", scaling = "density")

    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.vlines(x = [1, 3], ymin = 1e-5, ymax = 1e1, 
               colors = "red", linestyles = "dashed", alpha = 0.5)
    plt.xlim([0, 20])
    plt.ylim([1e-5, 1e1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    plt.savefig("./figures/periodogram_basic.pdf", bbox_inches="tight")
    # plt.show()


def main():
    seaborn.set_theme()
    np.random.seed(28)

    n, t0, tn = 500, 0, 4
    t = np.linspace(t0, tn, n).reshape(-1, 1)
    y = underlying(t[:,0]) + np.random.normal(0, 0.3, n)

    # plot_non_stationary_time_series(t, y)
    # plot_linear_trend(t, y)
    # plot_cubic_splines(t, y)
    plot_periodogram(t, y)

if __name__ == "__main__":
    main()
