import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import (
        periodogram, welch, spectrogram
)



def underlying(t):
    """ Underlying time series """
    return np.sin(2*np.pi*t) + 1/4*np.cos(6*np.pi*t) + t / 3


def non_stationary_func(t):
    """
    Example function which has frequencies on multiple scales,
    which changes over time
    """

    return np.where(t < 2, 3*np.sin(500*2*np.pi*t) + 2*np.sin(30*2*np.pi*t) + np.sin(100*2*np.pi*t),
               np.sin(300*2*np.pi*t) + 2*np.sin(50*2*np.pi*t) + 3*np.sin(700*2*np.pi*t))


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
    # plt.savefig("./figures/periodogram_basic.pdf", bbox_inches="tight")
    plt.show()


def plot_smoothed_periodogram(t, y):
    """
    Plot the smoothed periodogram, i.e., the estimated 
    power spectral density using a spectral window.
    """
    fs = 1/(t[1] - t[0])
    f, Pxx_den = welch(y ,fs, window = "bartlett", noverlap = 0,
                             nperseg = 256, detrend = "linear",  
                             scaling = "density")

    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.vlines(x = [1, 3], ymin = 1e-5, ymax = 1e1, 
               colors = "red", linestyles = "dashed", alpha = 0.5)
    plt.xlim([0, 20])
    plt.ylim([1e-5, 1e1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    plt.savefig("./figures/periodogram_smoothed.pdf", bbox_inches="tight")
    plt.show()


def plot_windows():
    """
    Plot the hann window, both the lag window
    and the spectral window.
    """
    M = 5 
    k = np.linspace(-6, 6, 500)
    w = np.linspace(-np.pi - 0.5, np.pi + 0.5, 500)
    def w_r(w):
        """ Rectangular spectral window """
        return 1/(2*np.pi)*np.sin(w*(M+0.5))/np.sin(w/2)
    def w_t(w):
        """ Hann spectral window """
        return (0.25*w_r(w - np.pi/M) + (1 - 0.5)*w_r(w) +
                0.25*w_r(w + np.pi/M))
    def w_h(k):
        """ Hann lag window """
        return np.where(np.abs(k) <= M, 0.5*(1 + np.cos(np.pi*k/M)), 0)
        

    plt.figure(figsize = (12, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(k, w_h(k), c = "k", label = r'$W_n^H(k)$')
    plt.xlabel(r'$k$')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(w, w_t(w), c = "k", label = r'$\mathcal{W}_n^H(\omega)$')
    plt.xlabel(r'$\omega$')
    plt.legend()
    plt.savefig("./figures/windows.pdf", bbox_inches = "tight")
    # plt.show()


def spectrogram_example():
    """
    Illustrate the difference in the periodogram, STFT and CWT
    """

    # Generate sample time series
    t = np.linspace(0, 4, 10000)
    y = non_stationary_func(t)
    
    # Compute periodogram
    fs = 1/(t[1] - t[0])
    f, Pxx_den = periodogram(y ,fs, window = "hann",
                             scaling = "density")
    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    plt.show()

    # Compute short-time Fourier transform and plot spectrogram
    fo, time, Sxx = spectrogram(y, fs, window = "hann")
    plt.pcolormesh(time, fo, Sxx)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time")
    plt.show()
   

    
    

def main():
    seaborn.set_theme()
    np.random.seed(28)

    n, t0, tn = 500, 0, 4
    t = np.linspace(t0, tn, n).reshape(-1, 1)
    y = underlying(t[:,0]) + np.random.normal(0, 0.3, n)

    # plot_non_stationary_time_series(t, y)
    # plot_linear_trend(t, y)
    # plot_cubic_splines(t, y)
    # plot_periodogram(t, y)
    # plot_smoothed_periodogram(t, y)
    # plot_windows()
    spectrogram_example()

if __name__ == "__main__":
    main()
