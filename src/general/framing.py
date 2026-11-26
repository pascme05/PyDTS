#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         framing
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
from src.external.sliding_window import sliding_window

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def idBlocks(ids, n):
    """Start and end rows of the contiguous ID blocks of a series with n rows (one block if ids is None)."""
    if ids is None:
        return [(0, n)]
    ids = np.asarray(ids)[:n]
    cut = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]

    return [(int(cut[k]), int(cut[k + 1])) for k in range(len(cut) - 1)]


def frameBlock(data, window, step):
    """Frames of one contiguous block: (N, window) for one column, (N, window, F) for several columns."""
    if data.ndim == 1:
        return sliding_window(data, window, step)

    return np.stack([sliding_window(np.ascontiguousarray(data[:, i]), window, step) for i in range(data.shape[1])],
                    axis=2)


def framing(data, window, overlap, ids=None):
    """
    Frames data (DataFrame, T x F) into windows of length window with the given overlap. If ids (one ID per row) is
    given, every contiguous ID block is framed separately, so that no window spans two IDs (blocks shorter than one
    window give no frames).
    """
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Start Framing Data")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    values = np.asarray(data.values if hasattr(data, 'values') else data, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    step = window - overlap

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    blocks = [(a, b) for a, b in idBlocks(ids, len(values)) if b - a >= window]
    if not blocks:
        raise SystemExit("ERROR: No ID block is longer than the window (%d samples)" % window)
    if len(blocks) < len(idBlocks(ids, len(values))):
        print("WARN: %d ID block(s) shorter than the window are skipped" % (len(idBlocks(ids, len(values))) - len(blocks)))
    out = np.concatenate([frameBlock(values[a:b], window, step) for a, b in blocks], axis=0)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


def frameIds(ids, n, window, overlap):
    """ID of every frame produced by framing(..., ids) for a series with n rows."""
    step = window - overlap
    if ids is None:
        return None
    ids = np.asarray(ids)[:n]
    out = [np.full(int(np.floor((b - a - window) / step)) + 1, ids[a]) for a, b in idBlocks(ids, n) if b - a >= window]

    return np.concatenate(out)
