#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         help.py
# Date:         25.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function holds the extended help of the parameters (shown in the question marks of the forms), the parameters of
the basic (non-expert) mode and the conditions under which a parameter or category is used at all. The texts are
written from the code that reads each parameter (src/…). Keys are 'Section.variable' (the sections of PyDTS are flat).

HELP:            'Section.variable' -> markdown text (what it does, effect of the value, typical values)
BASIC:           parameters shown when the expert mode is off
ACTIVE:          'Section.variable' -> function(cfg) that is False if the parameter has no effect with the current
                 settings (hidden in the basic mode, marked as inactive in the expert mode)
RELEVANT:        section -> {category: function(cfg)} False if the category is not used by the selected solver/model
CATEGORY_TITLES: category -> title of its expander
"""

#######################################################################################################################
# Help texts
#######################################################################################################################
HELP = {
    # ==================================================================================================================
    # Experiment (start.py: setupExp)
    # ==================================================================================================================
    'Exp.name': "Name of the simulation. The trained model is stored in `mdl/mdl_<model>_<name>/` and loaded again "
                "for testing when *Training* = 0 — keep the name to test a model that was trained before.",
    'Exp.method':
        "How the data are split into training, validation and test data (src/data/loadData.py):\n"
        "1. **1-fold**: the first training dataset is split in time: the first *Training share* for training (of "
        "which the last *Validation share* … for validation), the rest for testing.\n"
        "2. **k-fold**: the first training dataset is split into *k* folds, every fold is tested once; the errors "
        "are averaged.\n"
        "3. **Transfer**: training on all *Training datasets*, validation on the *Validation dataset*, testing on "
        "the *Test dataset* — the realistic test of generalisation to a new operating cycle.\n"
        "4. **ID based**: rows with the *Test IDs* / *Validation IDs* (column `id` of the data) are held out.",
    'Exp.trainBatch': "0: all training windows at once (fastest, needs the memory). 1: the data are cut into blocks "
                      "of *Data › batch* samples that are trained one after the other. 2: one block per ID.",
    'Exp.kfold': "Number of folds of the k-fold cross validation. More folds use more data for training per fold, "
                 "the run time grows linearly. Typical 5 – 10.",
    'Exp.train': "0: no training — the model `mdl/mdl_<model>_<name>` must exist. 1: a new model is trained; if the "
                 "folder already holds weights, training continues from them (delete the folder to start fresh).",
    'Exp.test': "1: the test data are predicted, the errors computed (MAE, RMSE, R², …) and shown under Results.",
    'Exp.save': "1: PyDTS writes predictions and errors to `results/` (src/general/save.py). The GUI stores its own "
                "copy of every run in `results/gui/` anyway.",
    'Exp.plot': "1: the matplotlib figures of PyDTS are created (Results › Figures & files). The interactive charts "
                "of the GUI do not need them; switch off for faster runs.",
    'Exp.gpu': "0: CPU. 1: first CUDA GPU (PyTorch with CUDA; falls back to the CPU if none is found). On CPU, "
               "an LSTM with a long "
               "window trains slowly — reduce *Epochs* for first tests.",
    'Exp.log': "No effect since the PyTorch version: every epoch is written to `training_log.csv` of the "
               "model folder (shown under Results › Training).",
    'Exp.warn': "Console messages of the libraries (TF_CPP_MIN_LOG_LEVEL, kept for compatibility). 3 keeps the "
                "console readable.",
    'Exp.seed': "Random seed of the weight initialisation, dropout and shuffle order of the neural networks. The same "
                "seed and settings give the same model on the same machine (deterministic GPU kernels). Different "
                "seeds can give clearly different models when the test data lie outside the training data (Tutorial "
                "6: the cold drive ID 37, flagged by the range warning of the test; one seed gave R² below 0, others "
                "0.98–0.99). Use *Ensemble size* 3–5 instead of trusting one seed.",
    'Exp.ensemble': "Seed ensemble (DL and CL): N networks are trained with the seeds *seed*, *seed*+1, ... and stored "
                    "as model.pt, model_2.pt, ...; the test predicts with every network and uses the mean. The console "
                    "lists the errors of every member, their mean ± standard deviation (how much a single run depends "
                    "on the seed) and the errors of the mean prediction. 1: one network. 3–5 makes results robust "
                    "against an unlucky seed at 3–5 times the training time.",
    # ==================================================================================================================
    # Data (setupDat)
    # ==================================================================================================================
    'Dat.type': "File type of the datasets in `data/`. xlsx: sheets `input`, `output` and `units`; mat: "
                "variables `input`, `output`, `labelInp`, `labelOut`, `unitInp`, `unitOut`. Both need the columns "
                "`time` and `id`.",
    'Dat.train': "Names of the training datasets in `data/` without extension, comma separated. For 1-fold, k-fold "
                 "and ID based splits only the **first** one is used (training *and* testing data come from it).",
    'Dat.test': "Test dataset (transfer split only). Use a measurement the model has never seen, e.g. another drive "
                "cycle of the motor.",
    'Dat.val': "Validation dataset (transfer split only): decides early stopping and the learning-rate reduction.",
    'Dat.inp': "Input channels (X), comma separated; empty: all input columns. For temperature prediction the "
               "quantities that cause the losses (currents, voltages, speed, torque) and the boundary temperatures "
               "(ambient, coolant).",
    'Dat.out': "Output channels (y) that are predicted, comma separated (e.g. `T_sw, T_st` for winding and tooth "
               "temperature). One network predicts all outputs at once.",
    'Dat.fs': "Sampling frequency of the data in Hz (1/900 for 15-min data, 1 for 1 s). Used for the time axes and "
              "to convert the window length into a time span: *window* / *fs*.",
    'Dat.resample': "How the data are brought to *fs* (each ID on its own). hold: the last raw sample at or before each "
                    "new time step (zero-order hold). mean: the mean of the raw samples in the interval before each "
                    "new time step; when *fs* is lower than the raw rate this suppresses aliasing of fast signals "
                    "(currents, torque) and keeps their average, e.g. the mean losses that heat a machine. Same result "
                    "as hold when *fs* is at least the raw rate.",
    'Dat.lim': "0: all samples. x: only the first x samples (quick tests).",
    'Dat.rT': "1-fold split: share of the dataset used for training (and validation); the rest (at the end in time) "
              "is the test data. Typical 0.7 – 0.9.",
    'Dat.rV': "Share of the training data held out for validation (early stopping, learning-rate reduction); "
              "it is the last part of the training data in time and is not used for training. With the ID "
              "split it is the last share of each validation ID (or of every training ID). Typical 0.1 – 0.2.",
    'Dat.Shuffle': "Shuffling when splitting. **Keep False for time series** whose test data must be a later, "
                   "unseen period — shuffled windows overlap with training windows and give over-optimistic errors. "
                   "ID: whole blocks (IDs) are shuffled, the order inside a block is kept.",
    'Dat.inpNorm': "Normalisation of the inputs. Networks train best with inputs of similar size: mean/σ (3) is the "
                   "usual choice; 0 … 1 (2) for bounded signals.",
    'Dat.outNorm': "Normalisation of the outputs; the predictions are scaled back before the errors are computed, so "
                   "the errors are in the unit of the output.",
    'Dat.inpNoise': "Gaussian noise in % added to the inputs (data augmentation / robustness tests). 0: none.",
    'Dat.outNoise': "Gaussian noise in % added to the outputs. 0: none.",
    'Dat.inpFil': "Median filter of the inputs (removes spikes).", 'Dat.outFil': "Median filter of the outputs.",
    'Dat.threshold': "Outputs above this value count as 'on' for the state metrics ACC and F1 (energy "
                     "disaggregation, anomaly detection). Irrelevant for temperature regression.",
    'Dat.balance': "Sample weights of the training (DL and CL solvers). 0: none; 1: balanced classes "
                   "(classification); x > 1: the first output is split into x equally wide bins and every bin gets "
                   "the same total weight (rare high temperatures count as much as frequent low ones); -1: "
                   "window based balancing.",
    'Dat.batch': "Block size in samples for *Training batches* = 1.",
    'Dat.idT': "IDs (column `id`) held out for testing with the ID based split, comma separated.",
    'Dat.idV': "IDs that provide the validation data with the ID based split: the last share Data › rV of each "
               "of them is held out, the rest stays in the training data, so every non-test ID is trained on. "
               "If empty or only test IDs, the last share rV of every training ID is used.",
    # ==================================================================================================================
    # General parameters (setupPar)
    # ==================================================================================================================
    'Par.method': "0: regression (linear output layer) — e.g. temperatures. 1: classification (sigmoid output).",
    'Par.solver':
        "Family of models:\n"
        "- **SP** signal processing: state-space (SS) or transfer-function (TF) identification — linear, few "
        "parameters, a good baseline for thermal systems (which are nearly linear RC networks).\n"
        "- **ML** machine learning: random forest, KNN, SVM on frames or features.\n"
        "- **DL** deep learning: DNN, CNN, LSTM, transformer … on windows of the inputs.\n"
        "- **SF** short-term forecasting (NeuralForecast: LSTM, N-HiTS, N-BEATS).\n"
        "- **CL** closed loop: the prediction is fed back as input (src/model/trainMdlCL).",
    'Par.model': "Model of the solver. DL: **DNN** flattens the window (fast, no time order), **CNN** learns local "
                 "filters (transients), **LSTM** carries a memory state (slow heating), **TRAN** self-attention. In "
                 "the PyDTS journal the CNN gave the best motor temperatures. ML: RF, KNN, SVM. SP: SS, TF.",
    'Par.window': "Window length in samples: what the model sees of the past for one prediction. For temperatures "
                  "the window must cover the thermal time constant that matters (window / fs ≳ τ), or the history "
                  "must come in through rolled features (EWMA). Theory › Data & windows shows the time span. Windows "
                  "are cut per ID: no window spans two IDs, so the first window − 1 samples of every ID get no "
                  "prediction, and IDs shorter than one window are skipped.",
    'Par.overlap': "Overlap of consecutive training windows in samples. window − 1 gives one window per sample (most "
                   "training data, slow); smaller values thin the data. During testing PyDTS always uses window − 1 "
                   "(seq2point) unless −1.",
    'Par.outseq': "0: **seq2point** — one output sample per window (the *Focus point*). x > 0: **seq2seq** — the "
                  "network predicts x samples at once; testing then steps by x samples.",
    'Par.yFocus': "Sample of the window that is predicted in seq2point (0 … window − 1). window − 1: the last sample "
                  "(causal, the model can run on-line); window/2: the middle (uses future inputs, smoother). −1: the "
                  "mean of the window.",
    'Par.frame': "1: sliding windows (needed for CNN/LSTM/DNN on sequences). 0: sample by sample (static "
                 "regression, e.g. RF on the current values).",
    'Par.feat': "0: raw values. 1: statistical features per window (mean, std, RMS, … see Model › Features). 2: rolled "
                "features per input (EWMA/EWMS — the thermal memory as low-pass filters). 3: both.",
    'Par.init': "1: the initial values of y are added as an input (helps when the starting temperature is unknown "
                "to the model).",
    'Par.nDim': "3: windows as (samples × time × features) for CNN/LSTM. 2: flattened (samples × features) for "
                "DNN/ML.",
    'Par.lag': "Shift between input and output in samples (delay of the system).",
    'Par.ahead': "Forecast horizon of the SF solver in samples.",
    'Par.rank': "Feature ranking after testing (src/general/featureRank.py): 1 importance of every input, 2 "
                "correlation input–output, 3 input–input. Shown under Results › Features.",
    'Par.outMin': "Predictions are clipped to [min, max] (e.g. 0 for powers that cannot be negative).",
    'Par.outMax': "Upper clip of the predictions.",
    # ==================================================================================================================
    # Model parameters (mdlPara.py: setupMdl)
    # ==================================================================================================================
    'Mdl.batch': "Windows per gradient step. Larger batches train faster per epoch and smoother but generalise "
                 "slightly worse. Typical 64 – 1000. The number of steps per epoch is training windows / batch.",
    'Mdl.epoch': "Maximum passes over the training data; early stopping ends earlier when the validation loss does "
                 "not improve for *Patience* epochs. First tests: 5 – 20; final: 100+.",
    'Mdl.patience': "Epochs without improvement of the validation loss before training stops (the best weights are "
                    "restored). The learning rate is halved after patience/2 epochs without improvement.",
    'Mdl.valsteps': "Validation batches per epoch (validation loss = mean over them).",
    'Mdl.loss': "Loss that the training minimises. **mae**: robust to outliers, gives the median; **mse**: "
                "punishes large errors (hot-spot peaks) more. Cross entropy for classification.",
    'Mdl.metric': "Additional metric logged per epoch (does not influence the training).",
    'Mdl.opt': "Optimiser. **Adam** is the robust default; RMSprop for recurrent networks; SGD "
               "with momentum for fine tuning.",
    'Mdl.lr': "Learning rate: step size of the optimiser. Too large: the loss jumps or diverges; too small: slow. "
              "Adam: 1e-3 is the standard start; 1e-4 for fine tuning.",
    'Mdl.beta1': "Adam: decay of the running mean of the gradients (momentum). Default 0.9.",
    'Mdl.beta2': "Adam: decay of the running mean of the squared gradients. Default 0.999.",
    'Mdl.eps': "Small constant against division by zero in Adam/RMSprop.",
    'Mdl.rho': "RMSprop: decay of the squared-gradient history.", 'Mdl.mom': "Momentum of RMSprop/SGD.",
    'Mdl.verbose': "Console output during training: 0 = none, 1 or 2 = one line per epoch (loss, val_loss, learning rate).",
    'Mdl.shuffle': "Shuffle the training windows before every epoch (the windows keep their inner time order).",
    'Mdl.feat_roll.EWMA': "Spans s (samples) of exponentially weighted moving averages added as inputs (Features = 2 "
                          "or 3). An EWMA is a first-order low pass with time constant ≈ s/2 samples — like one RC "
                          "element of a Foster network. Choose spans that cover the thermal time constants, e.g. "
                          "`100, 1000, 5000` for 1-s data of a motor. 0: none.",
    'Mdl.feat_roll.EWMS': "Spans of exponentially weighted moving standard deviations (the fluctuation of the losses). "
                          "0: none.",
    'Mdl.feat_roll.diff': "1: the first difference of every input is added (rate of change).",
    'Mdl.SP_SS_order': "Order of the identified state-space model (number of states ≈ number of RC elements).",
    'Mdl.SP_SS_block': "Block rows of the Hankel matrices of the subspace identification (≥ order).",
    'Mdl.SK_RF_estimators': "Number of trees of the random forest.", 'Mdl.SK_RF_depth': "Maximum depth of a tree.",
    'Mdl.SK_KNN_neighbors': "Number of neighbours averaged by KNN.",
    'Mdl.SK_SVM_C': "Regularisation of the SVM (larger: fits the training data closer).",
    'Mdl.SK_SVM_gamma': "Width of the rbf kernel. 'scale' (default) adapts it to the number of inputs: 1 / (number of "
                        "features × variance). A fixed number must be about that small — a window of 96 samples × 3 "
                        "inputs has 288 features, so gamma = 0.1 makes every training window look unrelated to every "
                        "other one (start.py example: MAE 5787 MW with 0.1, 2332 MW with 'scale').",
}

#######################################################################################################################
# Basic parameters (expert mode off)
#######################################################################################################################
BASIC = {
    'Exp.name', 'Exp.method', 'Exp.train', 'Exp.test', 'Exp.plot', 'Exp.kfold',
    'Dat.type', 'Dat.train', 'Dat.test', 'Dat.val', 'Dat.inp', 'Dat.out', 'Dat.fs', 'Dat.rT', 'Dat.rV', 'Dat.lim',
    'Dat.inpNorm', 'Dat.outNorm', 'Dat.idT', 'Dat.idV', 'Dat.resample', 'Exp.seed', 'Exp.ensemble',
    'Par.method', 'Par.solver', 'Par.model', 'Par.window', 'Par.overlap', 'Par.outseq', 'Par.yFocus', 'Par.feat',
    'Par.frame', 'Par.ahead', 'Par.rank',
    'Mdl.batch', 'Mdl.epoch', 'Mdl.patience', 'Mdl.loss', 'Mdl.opt', 'Mdl.lr', 'Mdl.feat_roll.EWMA',
    'Mdl.feat_roll.EWMS', 'Mdl.SP_SS_order', 'Mdl.SP_TF_poles', 'Mdl.SP_TF_zeros', 'Mdl.SK_RF_estimators',
    'Mdl.SK_RF_depth', 'Mdl.SK_KNN_neighbors', 'Mdl.SK_SVM_kernel', 'Mdl.SK_SVM_C',
}


#######################################################################################################################
# Activity rules
#######################################################################################################################
def _solver(c):
    return str(c['Par'].get('solver'))


def _keras(c):
    return _solver(c) in ('DL', 'CL', 'SF')


def _opt(c, *names):
    return _keras(c) and str(c['Mdl'].get('opt')) in names


ACTIVE = {
    'Exp.kfold': lambda c: c['Exp'].get('method') == 1,
    'Dat.test': lambda c: c['Exp'].get('method') == 2, 'Dat.val': lambda c: c['Exp'].get('method') == 2,
    'Dat.idT': lambda c: c['Exp'].get('method') == 3, 'Dat.idV': lambda c: c['Exp'].get('method') == 3,
    'Dat.rT': lambda c: c['Exp'].get('method') == 0, 'Dat.rV': lambda c: c['Exp'].get('method') in (0, 1, 3),
    'Dat.batch': lambda c: c['Exp'].get('trainBatch') == 1,
    'Dat.inpFilLen': lambda c: c['Dat'].get('inpFil') == 1, 'Dat.outFilLen': lambda c: c['Dat'].get('outFil') == 1,
    'Dat.his': lambda c: _solver(c) == 'SF', 'Dat.fut': lambda c: _solver(c) == 'SF',
    'Par.ahead': lambda c: _solver(c) == 'SF',
    'Par.window': lambda c: c['Par'].get('frame') == 1, 'Par.overlap': lambda c: c['Par'].get('frame') == 1,
    'Par.outseq': lambda c: c['Par'].get('frame') == 1,
    'Par.yFocus': lambda c: c['Par'].get('frame') == 1 and not c['Par'].get('outseq'),
    'Mdl.batch': _keras, 'Mdl.epoch': _keras, 'Mdl.patience': _keras, 'Mdl.valsteps': _keras,
    'Mdl.loss': _keras, 'Mdl.metric': _keras, 'Mdl.opt': _keras, 'Mdl.lr': _keras, 'Mdl.verbose': _keras,
    'Mdl.shuffle': _keras,
    'Mdl.beta1': lambda c: _opt(c, 'Adam'), 'Mdl.beta2': lambda c: _opt(c, 'Adam'),
    'Mdl.eps': lambda c: _opt(c, 'Adam', 'RMSprop'), 'Mdl.rho': lambda c: _opt(c, 'RMSprop'),
    'Mdl.mom': lambda c: _opt(c, 'RMSprop', 'SDG'),
    'Mdl.feat_roll.EWMA': lambda c: c['Par'].get('feat') in (2, 3),
    'Mdl.feat_roll.EWMS': lambda c: c['Par'].get('feat') in (2, 3),
    'Mdl.feat_roll.diff': lambda c: c['Par'].get('feat') in (2, 3),
    'Mdl.SF_Freq': lambda c: _solver(c) == 'SF',
    'Exp.ensemble': lambda c: _solver(c) in ('DL', 'CL'),
}
for _f in ('Mean', 'Std', 'RMS', 'Peak2Rms', 'Median', 'Min', 'Max', 'Per25', 'Per75', 'Energy', 'Var', 'Range',
           '3rdMoment', '4thMoment', 'fft'):
    ACTIVE[f"Mdl.feat.{_f}"] = lambda c: c['Par'].get('feat') in (1, 3)

#######################################################################################################################
# Categories
#######################################################################################################################
RELEVANT = {
    'Mdl': {'SS': lambda c: _solver(c) == 'SP' and c['Par'].get('model') == 'SS',
            'TF': lambda c: _solver(c) == 'SP' and c['Par'].get('model') == 'TF',
            'RF': lambda c: _solver(c) == 'ML' and c['Par'].get('model') == 'RF',
            'SVM': lambda c: _solver(c) == 'ML' and c['Par'].get('model') == 'SVM',
            'KNN': lambda c: _solver(c) == 'ML' and c['Par'].get('model') == 'KNN',
            'Short Term Time-Series Forecasting (SF) Models': lambda c: _solver(c) == 'SF',
            'Hyperparameters': _keras, 'Solver Parameters': _keras,
            'Features (frame based)': lambda c: c['Par'].get('feat') in (1, 3),
            'Features (input based)': lambda c: c['Par'].get('feat') in (2, 3)},
}
CATEGORY_TITLES = {'SS': 'Signal processing · state space', 'TF': 'Signal processing · transfer function',
                   'RF': 'Machine learning · random forest', 'SVM': 'Machine learning · SVM',
                   'KNN': 'Machine learning · KNN', 'Hyperparameters': 'Training',
                   'Solver Parameters': 'Loss and optimiser', 'Training/Testing': 'Training and testing',
                   'Input/ Output Mapping': 'Inputs and outputs',
                   'Short Term Time-Series Forecasting (SF) Models': 'Short-term forecasting'}


#######################################################################################################################
# Helper
#######################################################################################################################
def key(section, p):
    return f"{section}.{p['var']}"


def is_basic(section, p):
    return not BASIC or key(section, p) in BASIC


def is_active(section, p, cfg):
    rule = ACTIVE.get(key(section, p))
    if rule is None:
        return True
    try:
        return bool(rule(cfg))
    except (KeyError, TypeError, ValueError):
        return True
