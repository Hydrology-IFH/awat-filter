# please ensure that the required packages are installed
# use reticulate to run filter from R
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm


def awat_filter(y, w_max, d_max, k_max=6, w_min=1, d_min=0.01):
    """Apply an adaptive window and adaptive threshold (AWAT) filter to an
    array.

    Args
    ----------
    y : array_like, numpy.array
        data with numeric values (i.e. no NaN or inf values)

    w_max : int
        maximum window length (i.e. number of observations)

    d_max : int
        maximum threshold value (in mm)

    k_max : int, optional
        maximum polynomial order used to fit the polynomials. Default is 6.

    w_min : int, optional
        minimum window length (i.e. number of obervations). Default is 1.

    d_min : int, optional
        minimum threshold value (in mm). depends on scale accuracy. Default
        is 0.01.

    Reference
    ----------
    Peters, A., Nehls, T., Schonsky, H., and Wessolek, G.: Separating
    precipitation and evapotranspiration from noise: a new filter routine for
    high-resolution lysimeter data, Hydrol. Earth Syst. Sci., 18, 1189-1198,
    10.5194/hess-18-1189-2014, 2014.

    Returns
    ----------
    y_filt : np.ndarray
        filtered data
    """
    w_min = 1
    if k_max > w_max:
        k_max = w_max - 2
    win_half = int(np.floor(w_max/2))

    y = np.asarray(y)
    # Ensure that x is either single or double precision floating point.
    if y.dtype != np.float64 and y.dtype != np.float32:
        y = y.astype(np.float64)

    if w_max > y.size:
        raise ValueError("w_max must be less "
                         "than or equal to the size of y.")

    if np.isnan(y).any():
        raise ValueError("Data contains non-numeric values.")

    # select optimal order k for moving polynomials
    k_arr = np.ones(len(y))

    # loop over windows
    pbar = tqdm(total=len(y) - w_max)  # progress bar
    for t in range(0 + win_half, len(y) - win_half):
        # select window
        y_win = y[t-win_half:t+win_half+1]
        aicc_arr = np.zeros(k_max)
        # loop over k
        for k in range(1, k_max+1):
            x_win = np.arange(len(y_win))
            # fit polynomial
            coeffs = np.polyfit(x_win, y_win, k)
            polyk = np.poly1d(coeffs)
            # make prediction
            y_win_hat = polyk(x_win)

            # calculate AIC
            resid_win = y_win - y_win_hat
            ssq_win = np.sum(np.square(resid_win))
            r_win = len(y_win)
            n_par = k
            aicc_arr[k-1] = r_win * np.log(ssq_win/r_win) + 2 * r_win + (2 * r_win * (r_win + 1)) / (r_win - n_par - 1)

        # select polynomial with smallest AIC
        k_arr[t] = np.argmin(aicc_arr) + 1
        pbar.update(1)

    # calculate signal strength and noise of moving polynomial
    b_arr = np.zeros_like(y)  # signal strength
    sres_arr = np.zeros_like(y)  # predicted noise
    sdat_arr = np.zeros_like(y)  # observed noise
    # loop over time steps
    for t in range(0 + win_half, len(y) - win_half):
        # select window
        y_win = y[t-win_half:t+win_half+1]
        r_win = len(y_win)
        x_win = np.arange(len(y_win))

        # fit polynomials
        coeffs = np.polyfit(x_win, y_win, k_arr[t])
        polyk = np.poly1d(coeffs)
        # make prediction
        y_win_hat = polyk(x_win)

        # calculate the noise
        resid_win = y_win - y_win_hat
        # calculate the noise
        sres_i = np.square((1/r_win) * np.sum(resid_win**2))
        y_win_mean = np.mean(y_win)
        diff_mean = y_win - y_win_mean
        sdat_i = np.square((1/r_win) * np.sum(diff_mean**2))
        sres_arr[t] = sres_i
        sdat_arr[t] = sdat_i
        # calculate the signal strength
        b_arr[t] = sres_i / sdat_i

    b_arr = np.nan_to_num(b_arr, posinf=1)

    # calculate the adaptive width of the moving window
    w_arr = np.zeros((len(y), 2))
    w_arr[:, 0] = w_min
    w_arr[:, 1] = b_arr * w_max
    # adaptive width of the moving window
    aw = np.max(w_arr, axis=1)
    aw[aw > w_max] = w_max

    # calculate the adaptive threshold value
    sres_mean = np.mean(sres_arr)
    sres_se = stats.sem(sres_arr)
    tval_975 = stats.t.interval(0.95, df=len(sres_arr)-1, loc=sres_mean, scale=sres_se)[-1]
    d_arr = np.zeros_like(y)
    cond_0 = (sres_arr * tval_975 >= d_max)
    cond_1 = (sres_arr * tval_975 > d_min) & (sres_arr * tval_975 < d_max)
    cond_2 = (sres_arr * tval_975 <= d_min)

    d_arr[cond_0] = d_max
    d_arr[cond_1] = sres_arr[cond_1] * tval_975
    d_arr[cond_2] = d_min

    # filter data
    y_filt = np.zeros_like(y)
    # loop over time steps
    for t in range(0 + win_half, len(y) - win_half):
        # apply moving average with adaptive window width
        win_half = int(np.floor(aw[t])/2)
        if win_half == 0:
            y_filt[t] = y[t]
        else:
            y_win = y[t-win_half:t+win_half+1]
            y_win_mean = np.mean(y_win)
            y_filt[t] = y_win_mean

    # apply adaptive threshold
    y_diff = np.abs(np.diff(y))
    cond_d = (y_diff < d_arr[1:])
    # discard values below measurement accuracy
    y_filt[1:][cond_d] = np.NaN
    y_filt[y_filt<=0.001] = np.NaN

    df_filt = pd.DataFrame(y_filt)
    df_filt = df_filt.fillna(method='ffill')
    y_filt = df_filt.values.flatten()

    return y_filt
