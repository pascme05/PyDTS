#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         start
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.external.zoh import zoh

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import pandas as pd


#######################################################################################################################
# Function
#######################################################################################################################
def resampleOne(data, fs, how='hold'):
    """
    Resampling of one contiguous record (one ID) onto a grid with sampling rate fs. how = 'hold': last raw sample at
    or before each grid point (zero-order hold); 'mean': mean of the raw samples in the interval (t - 1/fs, t]
    (anti-aliasing when the rate is reduced, held value if the interval contains no sample). 'time' and 'id' are
    always held.
    """
    N = len(data)
    tr = data['time'].values
    t = tr[0] + np.arange(int(np.floor((tr[N - 1] - tr[0]) * fs + 1e-9)) + 1) / fs                                       # step exactly 1/fs
    out = np.zeros((len(t), len(data.columns)))
    if how == 'mean':
        hi = np.searchsorted(tr, t + 1e-9 / fs, side='right')                                                          # samples <= t
        lo = np.searchsorted(tr, t - 1 / fs + 1e-9 / fs, side='right')                                                 # samples <= t - 1/fs
        lo[0] = 0
        cnt = hi - lo
    for i, name in enumerate(data.columns):
        out[:, i] = zoh(t, tr, data[name].values)
        if how == 'mean' and name not in ('time', 'id'):
            cs = np.r_[0, np.cumsum(data[name].values.astype(float))]
            m = cnt > 0
            out[m, i] = (cs[hi[m]] - cs[lo[m]]) / cnt[m]

    return [pd.DataFrame(out, columns=data.columns), t]


def resampleData(data, fs_raw, setupDat):
    """
    Resamples the data to setupDat['fs'] (setupDat['resample']: 'hold' zero-order hold (default) or 'mean' interval
    mean). Every ID is resampled on its own time grid, so that gaps
    between IDs (e.g. removed test IDs, or a time axis that restarts per ID) are not filled with held values.
    """
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    fs = setupDat['fs']
    how = str(setupDat.get('resample', 'hold'))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    if np.isclose(fs, fs_raw, rtol=1e-6):                                                                               # float time steps
        print("INFO: Sampling rate of the data is equivalent to target sampling rate %6.4f Hz" % fs)
        return [data, data['time'].values]

    if 'id' in data.columns:
        ids = data['id'].values
        start = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]                                            # contiguous ID blocks
        parts = [resampleOne(data.iloc[start[k]:start[k + 1]], fs, how) for k in range(len(start) - 1)]
        out = pd.concat([p[0] for p in parts], ignore_index=True)
        t = np.concatenate([p[1] for p in parts])
    else:
        [out, t] = resampleOne(data, fs, how)
    print("INFO: Sampling rate of the data %6.4f Hz is not equivalent to target sampling rate %6.4f Hz data"
          " is resampled (%s)" % (fs_raw, fs, how))

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [out, t]
