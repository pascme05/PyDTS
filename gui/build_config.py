#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         build_config.py
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
PyDTS has no config workbook: the parameters are set in start.py (setupExp, setupDat, setupPar) and mdlPara.py
(setupMdl). This script reads both files and writes the config workbook of the GUI (config/default.xlsx, sheets Exp,
Dat, Par, Mdl with the columns Name | Category | Description | Variable | Value | Unit), from which build_schema.py
creates the parameter forms. Categories are taken from the '# ----' headings of the scripts, descriptions from the
inline comments (rewritten below where options must be selectable). Afterwards the schema is built and the setups of
setup/tutorial, setup/lecture and setup/journal are converted into ready-to-load JSON configs (config/<name>.json).
Usage:      python gui/build_config.py
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import ast
import json
import os
import re
import sys

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#######################################################################################################################
# Settings
#######################################################################################################################
SECTIONS = {'Exp': 'setupExp', 'Dat': 'setupDat', 'Par': 'setupPar', 'Mdl': 'setupMdl'}
SKIP = {('Exp', 'author')}                                            # not used by the simulation
# list parameters: stored as comma separated text in the workbook, cast to lists by the adapter
LISTS = {('Dat', 'train'), ('Dat', 'idT'), ('Dat', 'idV'), ('Dat', 'inp'), ('Dat', 'out'), ('Dat', 'his'),
         ('Dat', 'fut'), ('Mdl', 'feat_roll.EWMA'), ('Mdl', 'feat_roll.EWMS')}
# (section, var) -> (name, description with selectable options, unit); None keeps the comment of the script
TEXT = {
    ('Exp', 'name'): ("Simulation name", "Name of the simulation; the model is stored as mdl/mdl_<model>_<name>.", None),
    ('Exp', 'sim'): ("Operating mode", "0) simulation (train and/or test), 1) hyperparameter optimisation, 2) grid "
                                       "optimisation", None),
    ('Exp', 'gpu'): ("Device", "0) CPU, 1) GPU", None),
    ('Exp', 'warn'): ("Library messages", "0) all messages, 1) no INFO, 2) no INFO and WARN, 3) none", None),
    ('Exp', 'seed'): ("Random seed", "Seed of the weight initialisation (default 42)", None),
    ('Exp', 'ensemble'): ("Ensemble size", "Number of networks trained with the seeds seed, seed+1, ... (DL, CL); the "
                                           "test uses their mean prediction", None),
    ('Exp', 'method'): ("Data split", "0) 1-fold with data split, 1) k-fold cross validation, 2) transfer (separate "
                                      "datasets), 3) ID based", None),
    ('Exp', 'trainBatch'): ("Training batches", "0) all data at once, 1) fixed batch size (Data › batch), 2) one "
                                                "batch per ID", None),
    ('Exp', 'kfold'): ("Number of folds", "Number of folds of the k-fold cross validation (data split 1).", None),
    ('Exp', 'train'): ("Training", "0) no training (load the stored model), 1) train a new model (or retrain)", None),
    ('Exp', 'test'): ("Testing", "0) no testing, 1) testing", None),
    ('Exp', 'save'): ("Save results", "0) results are not saved, 1) results are saved", None),
    ('Exp', 'log'): ("TensorBoard log", "0) no logging, 1) TensorBoard log of the training in ./logs", None),
    ('Exp', 'plot'): ("Plots", "0) no plotting, 1) plotting", None),
    ('Dat', 'type'): ("File type", "(xlsx): Excel workbook, (mat): MATLAB file", None),
    ('Dat', 'batch'): ("Batch size of the data", "Number of samples fed at once to training (Training batches = 1).",
                       'samples'),
    ('Dat', 'Shuffle'): ("Shuffling", "(False): no shuffling, (True): shuffle samples when splitting, (ID): shuffle "
                                      "the IDs (blocks)", None),
    ('Dat', 'rT'): ("Training share", "Share of the data used for training (0 … 1), the rest for testing.", '-'),
    ('Dat', 'rV'): ("Validation share", "Share of the training data held out for validation (0 … 1), not used for training.", '-'),
    ('Dat', 'idT'): ("Test IDs", "IDs used for testing (data split 3), comma separated.", None),
    ('Dat', 'idV'): ("Validation IDs", "IDs whose last share rV is used for validation (data split 3), comma separated; "
                               "if empty or test IDs, the last share rV of every training ID is used.", None),
    ('Dat', 'train'): ("Training datasets", "Names of the training datasets in data/ (without extension), comma "
                                            "separated.", None),
    ('Dat', 'test'): ("Test dataset", "Name of the test dataset in data/ (transfer, data split 2).", None),
    ('Dat', 'val'): ("Validation dataset", "Name of the validation dataset in data/ (transfer, data split 2).", None),
    ('Dat', 'inp'): ("Inputs (X)", "Names of the input channels, comma separated; empty: all inputs.", None),
    ('Dat', 'out'): ("Outputs (y)", "Names of the output channels that are predicted, comma separated.", None),
    ('Dat', 'his'): ("Historic exogenous inputs", "Historic exogenous inputs of the short-term forecast (SF).", None),
    ('Dat', 'fut'): ("Future exogenous inputs", "Future exogenous inputs of the short-term forecast (SF).", None),
    ('Dat', 'fs'): ("Sampling frequency", "Sampling frequency of the data.", 'Hz'),
    ('Dat', 'resample'): ("Resampling", "(hold): last sample (zero-order hold), (mean): mean over each new sample "
                                        "interval", None),
    ('Dat', 'lim'): ("Sample limit", "0: data are not limited, x: only the first x samples are used.", 'samples'),
    ('Dat', 'weightNorm'): ("Normalisation scope", "0) separate per input/output channel, 1) weighted (common)",
                            None),
    ('Dat', 'inpNorm'): ("Input normalisation", "0) none, 1) -1 … +1, 2) 0 … 1, 3) mean / standard deviation", None),
    ('Dat', 'outNorm'): ("Output normalisation", "0) none, 1) -1 … +1, 2) 0 … 1, 3) mean / standard deviation", None),
    ('Dat', 'inpNoise'): ("Input noise", "Gaussian noise added to the inputs (0: none).", '%'),
    ('Dat', 'outNoise'): ("Output noise", "Gaussian noise added to the outputs (0: none).", '%'),
    ('Dat', 'inpFil'): ("Input filter", "0) none, 1) median filter", None),
    ('Dat', 'outFil'): ("Output filter", "0) none, 1) median filter", None),
    ('Dat', 'inpFilLen'): ("Input filter length", "Length of the median filter of the inputs.", 'samples'),
    ('Dat', 'outFilLen'): ("Output filter length", "Length of the median filter of the outputs.", 'samples'),
    ('Dat', 'threshold'): ("On/off threshold", "Threshold that turns the regression output into on/off states (ACC, "
                                               "F1); 0: no threshold.", None),
    ('Dat', 'balance'): ("Class balancing", "0: none, 1: balance the classes, x > 1: balance x bins of the output, -1: "
                                            "balancing function.", None),
    ('Par', 'method'): ("Task", "0) regression, 1) classification", None),
    ('Par', 'solver'): ("Solver", "(SP): signal processing, (ML): machine learning, (DL): deep learning, (SF): short-"
                                  "term forecasting, (CL): closed loop", None),
    ('Par', 'model'): ("Model", "(DNN): dense network, (CNN): convolutional network, (LSTM): long short-term memory, "
                                "(TRAN): transformer, (DAE): denoising autoencoder, (INF): informer-type attention, "
                                "(TFS): time/frequency/statistics fusion, (RF): random forest, (KNN): k-nearest "
                                "neighbours, (SVM): support vector machine, (SS): state space, (TF): transfer "
                                "function, (NHITS): N-HiTS, (NBEATS): N-BEATS", None),
    ('Par', 'lag'): ("Lag", "Lag between input (X) and output (y).", 'samples'),
    ('Par', 'ahead'): ("Forecast horizon", "Number of samples predicted ahead (SF).", 'samples'),
    ('Par', 'frame'): ("Framing", "0) no framing (sample by sample), 1) framing (sliding windows)", None),
    ('Par', 'feat'): ("Features", "0) raw values, 1) statistical features per frame, 2) rolled features per input, "
                                  "3) rolled and frame features", None),
    ('Par', 'init'): ("Initial values", "0) none, 1) add the initial values of y as inputs", None),
    ('Par', 'window'): ("Window length", "Length of one input window.", 'samples'),
    ('Par', 'overlap'): ("Overlap", "Overlap of consecutive windows (-1: no overlap during testing).", 'samples'),
    ('Par', 'outseq'): ("Output sequence", "0: seq2point (one output per window), x: length of the predicted "
                                           "sub-sequence (seq2seq).", 'samples'),
    ('Par', 'yFocus'): ("Focus point", "Sample of the window that is predicted in seq2point (-1: average).",
                        'sample'),
    ('Par', 'nDim'): ("Input dimension", "2) flattened window (samples × features), 3) window (samples × time × "
                                         "features)", None),
    ('Par', 'rank'): ("Feature ranking", "0) none, 1) feature ranking, 2) input-output ranking, 3) input-input "
                                         "ranking", None),
    ('Par', 'outMin'): ("Output minimum", "Lower limit of the predicted values.", None),
    ('Par', 'outMax'): ("Output maximum", "Upper limit of the predicted values.", None),
    ('Mdl', 'batch'): ("Batch size", "Number of windows per gradient step (training and testing).", None),
    ('Mdl', 'epoch'): ("Epochs", "Maximum number of passes over the training data.", None),
    ('Mdl', 'patience'): ("Patience", "Epochs without improvement of the validation loss before training stops "
                                      "(learning rate halves after patience/2).", None),
    ('Mdl', 'valsteps'): ("Validation steps", "Number of validation batches per epoch.", None),
    ('Mdl', 'shuffle'): ("Shuffle batches", "(False): keep the order, (True): shuffle before every epoch", None),
    ('Mdl', 'verbose'): ("Training output", "0) silent, 1) progress bar, 2) one line per epoch", None),
    ('Mdl', 'loss'): ("Loss function", "(mae): mean absolute error, (mse): mean squared error, "
                                       "(BinaryCrossentropy): binary cross entropy, (KLDivergence): Kullback-Leibler",
                      None),
    ('Mdl', 'metric'): ("Metric", "(mae): mean absolute error, (mse): mean squared error, (accuracy): accuracy", None),
    ('Mdl', 'opt'): ("Optimiser", "(Adam): Adam, (RMSprop): RMSprop, (SGD): stochastic gradient descent", None),
    ('Mdl', 'lr'): ("Learning rate", "Step size of the optimiser.", None),
    ('Mdl', 'beta1'): ("β1 (Adam)", "Decay of the first moment (mean of the gradients).", None),
    ('Mdl', 'beta2'): ("β2 (Adam)", "Decay of the second moment (squared gradients).", None),
    ('Mdl', 'eps'): ("ε", "Small constant for numerical stability.", None),
    ('Mdl', 'rho'): ("ρ (RMSprop)", "Discounting factor of the gradient history.", None),
    ('Mdl', 'mom'): ("Momentum", "Momentum of RMSprop and SGD.", None),
    ('Mdl', 'SK_SVM_kernel'): ("SVM kernel", "(linear): linear, (poly): polynomial, (rbf): radial basis, "
                                             "(sigmoid): sigmoid", None),
    ('Mdl', 'SP_TF_method'): ("TF estimation", "(fft): FFT based, (h2): H2 norm", None),
    ('Mdl', 'SF_Freq'): ("Data frequency", "Pandas offset alias of the data (e.g. H, 15min) for the forecasting "
                                           "models.", None),
    ('Mdl', 'feat_roll.EWMA'): ("EWMA spans", "Spans of the exponentially weighted moving averages (0: none), comma "
                                              "separated.", 'samples'),
    ('Mdl', 'feat_roll.EWMS'): ("EWMS spans", "Spans of the exponentially weighted moving standard deviations (0: "
                                              "none), comma separated.", 'samples'),
    ('Mdl', 'feat_roll.diff'): ("Differences", "0) no, 1) add the first difference of every input", None),
}
FEAT = {'Mean': 'mean', 'Std': 'standard deviation', 'RMS': 'RMS value', 'Peak2Rms': 'crest factor (peak / RMS)',
        'Median': 'median', 'Min': 'minimum', 'Max': 'maximum', 'Per25': '25th percentile', 'Per75': '75th percentile',
        'Energy': 'energy (sum of squares)', 'Var': 'variance', 'Range': 'range (max − min)',
        '3rdMoment': 'skewness (3rd moment)', '4thMoment': 'kurtosis (4th moment)', 'fft': 'FFT magnitudes'}
CAT_MDL = {'feat': 'Features (frame based)', 'feat_roll': 'Features (input based)'}

LINE = re.compile(r"^\s*(setupExp|setupDat|setupPar|setupMdl)\[(['\"])(\w+)\2\](?:\[(['\"])(\w+)\4\])?\s*=\s*"
                  r"(.+?)\s*(?:#\s*(.*))?$")
HEAD = re.compile(r"^\s*#\s*([A-Za-z][\w /()&\-,.]*?)\s*$")


#######################################################################################################################
# Parsing
#######################################################################################################################
def _value(txt):
    """Python literal or simple arithmetic (1/900, +1e9) of a setup line."""
    txt = txt.strip()
    try:
        return ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        return eval(compile(ast.parse(txt, mode='eval'), '<setup>', 'eval'), {'__builtins__': {}}, {})


def parse(filename):
    """[(section, category, var, value, comment)] of all setup assignments of a start script, in order."""
    out, cat, prev_rule = [], None, False
    with open(filename, encoding='utf-8') as f:
        lines = f.read().splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.fullmatch(r'#\s*[-=]{10,}', s):
            prev_rule = True
            continue
        m = HEAD.match(ln)
        if m and prev_rule and i + 1 < len(lines) and re.fullmatch(r'\s*#\s*[-=]{10,}', lines[i + 1]):
            cat = m.group(1).strip()
        prev_rule = False
        m = LINE.match(ln)
        if not m:
            continue
        sec = {v: k for k, v in SECTIONS.items()}[m.group(1)]
        var = m.group(3) + (f".{m.group(5)}" if m.group(5) else '')
        try:
            val = _value(m.group(6))
        except Exception:
            continue                                                  # e.g. setupMdl = mdlPara(setupMdl)
        if isinstance(val, dict):
            continue                                                  # initialisation of a sub dict
        c = CAT_MDL.get(m.group(3), cat) if sec == 'Mdl' and m.group(5) else cat
        out.append((sec, c, var, val, (m.group(7) or '').strip()))
    return out


def _cell(sec, var, val):
    """Workbook cell of a value (lists as comma separated text)."""
    if (sec, var) in LISTS:
        return ', '.join(str(v) for v in (val or []))
    if isinstance(val, bool):
        return str(val)
    return val


#######################################################################################################################
# Workbook
#######################################################################################################################
def write_workbook(rows, filename):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sec in SECTIONS:
        ws = wb.create_sheet(sec)
        ws.append(['Name', 'Category', 'Description', 'Variable', 'Value', 'Unit'])
        for s, cat, var, val, com in rows:
            if s != sec or (s, var) in SKIP:
                continue
            name, desc, unit = TEXT.get((s, var), (None, None, None))
            if var.startswith('feat.'):
                k = var.split('.', 1)[1]
                name, desc = f"{k} feature", f"0) off, 1) {FEAT.get(k, k)} of every window and input"
            ws.append([name or var, cat or sec, desc or com or var, var, _cell(s, var, val), unit or '-'])
        for col, w in zip('ABCDEF', (28, 26, 90, 20, 22, 10)):
            ws.column_dimensions[col].width = w
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wb.save(filename)


#######################################################################################################################
# Ready-to-load configs from the setup scripts
#######################################################################################################################
def script_config(default, rows):
    """Default config with the values of a setup script (lists cast like the adapter)."""
    from gui import adapter as A
    cfg = json.loads(json.dumps(default))
    for s, _, var, val, _ in rows:
        if s in cfg and (s, var) not in SKIP and var in cfg[s]:
            cfg[s][var] = A.to_cfg(s, var, val)
    return cfg


def main():
    rows = parse(os.path.join(ROOT, 'start.py')) + parse(os.path.join(ROOT, 'mdlPara.py'))
    write_workbook(rows, os.path.join(ROOT, 'config', 'default.xlsx'))
    print(f"INFO: config/default.xlsx written ({len(rows)} parameters)")

    from gui import build_schema as B
    from gui import adapter as A
    schema, default = B.build_schema(os.path.join(ROOT, A.CONFIG_WORKBOOK))
    with open(os.path.join(ROOT, 'gui', 'schema.json'), 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, 'config', 'default.json'), 'w', encoding='utf-8') as f:
        json.dump(default, f, indent=2, ensure_ascii=False)
    for sec, ps in schema.items():
        print(f"INFO: {sec:<4} {len(ps):>4} parameters ({sum(1 for p in ps if p['options'])} with options)")

    mdl_default = parse(os.path.join(ROOT, 'mdlPara.py'))
    for folder in ('tutorial', 'lecture', 'journal'):
        d = os.path.join(ROOT, 'setup', folder)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not fn.endswith('.py') or fn.startswith('mdlPara'):
                continue
            r = parse(os.path.join(d, fn))
            mdl = os.path.join(d, 'mdlParaJournal.py') if folder == 'journal' else None
            extra = parse(mdl) if mdl and os.path.isfile(mdl) else mdl_default
            cfg = script_config(default, extra + r)
            with open(os.path.join(ROOT, 'config', fn[:-3] + '.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps({'_meta': {'tool': 'PyDTS', 'format': 'config', 'version': 1,
                                              'source': f"setup/{folder}/{fn}"}, **cfg}, indent=2,
                                   ensure_ascii=False))
            print(f"INFO: config/{fn[:-3]}.json written from setup/{folder}/{fn}")


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == '__main__':
    main()
