#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         trainMdlDLopti
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Hyperparameter optimisation of the DL models with Optuna (TPE sampler, Hyperband pruning). The search space follows
the configured model (CNN, LSTM or DNN, see models.optiMdl). Every trial trains up to setupMdl['epoch'] epochs; the
best trial is retrained and stored in mdl/mdl_opt_<model>_<name>/ (model.pt, best_params.json, trials.csv).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.helpFnc import sampleWeights, toBool
from src.general.helpFnc import reshapeMdlData
from src.model.models import optiMdl
from src.model import torchFit as tfit

# ==============================================================================
# External
# ==============================================================================
import json
import os

import optuna


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlDLopti(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Optimising Model (DL)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dev = tfit.device(setupExp['gpu'])
    save_dir = os.path.join('mdl', 'mdl_opt_' + setupPar['model'] + '_' + setupExp['name'])
    os.makedirs(save_dir, exist_ok=True)
    nTrials = int(setupMdl.get('trials', 30))
    shuffle = toBool(setupMdl['shuffle'])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    weights = sampleWeights(data['T']['y'], setupDat['balance'])
    [data['T']['X'], data['T']['y']] = reshapeMdlData(data['T']['X'], data['T']['y'], setupDat, setupPar, 0)
    [data['V']['X'], data['V']['y']] = reshapeMdlData(data['V']['X'], data['V']['y'], setupDat, setupPar, 0)
    T, V = (data['T']['X'], data['T']['y']), (data['V']['X'], data['V']['y'])
    shape = data['T']['X'].shape[1:]
    out = tfit.outDim(data['T']['y'], setupDat, setupPar)
    activation = 'linear' if setupPar['method'] == 0 else 'sigmoid'

    ###################################################################################################################
    # Search
    ###################################################################################################################
    def objective(trial):
        mdl, lr = optiMdl(trial, setupPar['model'], shape, out, activation)
        hist = tfit.fit(mdl, T, V, setupMdl, None, None, dev, weights=weights, shuffle=shuffle, verbose=0, lr=lr,
                        log=False, trial=trial)
        best = min(h['val_loss'] for h in hist)
        print("INFO: Trial %d: val_loss %.4g after %d epochs, %s" % (trial.number, best, len(hist), trial.params),
              flush=True)
        return best

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                      max_resource=int(setupMdl['epoch']),
                                                                      reduction_factor=3))
    study.optimize(objective, n_trials=nTrials)

    ###################################################################################################################
    # Best model
    ###################################################################################################################
    print("INFO: Best trial %d: val_loss %.4g, %s" % (study.best_trial.number, study.best_value, study.best_params))
    mdl, lr = optiMdl(optuna.trial.FixedTrial(study.best_params), setupPar['model'], shape, out, activation)
    info = {'model': 'opt_' + setupPar['model'], 'shape': [int(s) for s in shape], 'out': int(out),
            'method': int(setupPar['method']), 'params': study.best_params}
    hist = tfit.fit(mdl, T, V, setupMdl, os.path.join(save_dir, 'model.pt'), info, dev, weights=weights,
                    shuffle=shuffle, verbose=setupMdl['verbose'], lr=lr)

    ###################################################################################################################
    # Output
    ###################################################################################################################
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        json.dump({'params': study.best_params, 'val_loss': study.best_value,
                   'retrained_val_loss': min(h['val_loss'] for h in hist)}, f, indent=2)
    study.trials_dataframe().to_csv(os.path.join(save_dir, 'trials.csv'), index=False)
    print(mdl)
    print("INFO: Best epoch: %d" % (min(range(len(hist)), key=lambda i: hist[i]['val_loss']) + 1))
    print("INFO: Model size (kB): %.2f" % tfit.mdlSize(mdl))
