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
def resampleData(data, fs_raw, setupDat):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    fields = data.columns
    fs = setupDat['fs']
    N = len(data)

    # ==============================================================================
    # Variables
    # ==============================================================================
    t = np.linspace(data['time'].iloc[0], data['time'].iloc[N - 1],
                    int(np.floor((data['time'].iloc[N - 1] - data['time'].iloc[0]) * fs) + 1))
    out = np.zeros((len(t), len(fields)))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Interpolate
    # ==============================================================================
    if fs == fs_raw:
        print("INFO: Sampling rate of the data is equivalent to target sampling rate %6.4f Hz" % fs)
        out = data
    else:
        # ------------------------------------------
        # Calc
        # ------------------------------------------
        for i in range(0, len(fields)):
            out[:, i] = zoh(t, data['time'], data[fields[i]])

        # ------------------------------------------
        # Out
        # ------------------------------------------
        out = pd.DataFrame(out, columns=fields)

        # ------------------------------------------
        # Msg
        # ------------------------------------------
        print("INFO: Sampling rate of the data %6.4f Hz is not equivalent to target sampling rate %6.4f Hz data"
              " is resampled" % (fs_raw, fs))

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [out, t]