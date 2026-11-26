#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         save
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

# ==============================================================================
# External
# ==============================================================================
import os
import sys
import csv
import contextlib
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from src.general.printResults import printResults


#######################################################################################################################
# Function
#######################################################################################################################
def save(YTest, YPred, resultsApp, resultsAvg, setupDat, setupPar, setupExp, setupPath):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Saving results")

    ####################################################################################################################
    # Init
    ####################################################################################################################
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ####################################################################################################################
    # Directory
    ####################################################################################################################
    if setupExp['method'] == 1:
        if setupExp['sim'] == 0:
            org_string = setupExp['name']
        else:
            org_string = setupExp['nameInit']
        split_string = org_string.split("kfold", 1)
        dir_name = split_string[0] + 'kfold' + str(setupDat['fold'])
    else:
        if setupExp['sim'] == 0:
            dir_name = setupExp['name']
        else:
            dir_name = setupExp['nameInit']
    path = os.path.join(setupPath['resPath'], dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # Save Setup
    ####################################################################################################################
    if setupExp['sim'] == 0:
        setup = {'Exp': setupExp, 'Par': setupPar,
                 'Dat': {k: v for k, v in setupDat.items() if not k.startswith(('norm', 'inp', 'out', 'Ts_', 'fs_'))
                         or k in ('inp', 'out', 'inpNorm', 'outNorm', 'inpFil', 'outFil', 'inpFilLen', 'outFilLen',
                                  'inpNoise', 'outNoise')}}
        with open(os.path.join(path, 'setup_' + dir_name + '.txt'), 'w') as file:
            file.write(json.dumps(setup, indent=2, default=str))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    if setupExp['sim'] == 0:
        resultName = os.path.join(path, 'result_' + setupExp['name'] + '_' + dt_string + '.txt')
        with open(resultName, "w") as file, contextlib.redirect_stdout(file):
            printResults(resultsApp, resultsAvg, setupDat)
    elif setupExp['sim'] == 2:
        resultName = os.path.join(path, 'result_Grid_Perf_' + setupExp['nameInit'] + '.csv')
        with open(resultName, 'a+', newline='') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
            if int(setupPar['optACC']) == 9:
                writer.writerow([setupPar['window'], setupDat['fs']] + list(resultsAvg))
            else:
                writer.writerow([setupPar['window'], setupDat['fs'], resultsAvg[int(setupPar['optACC'])]])

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    if setupExp['sim'] == 0:
        grtName = os.path.join(path, 'grt_' + setupExp['name'] + '_' + dt_string + '.csv')
        prdName = os.path.join(path, 'prd_' + setupExp['name'] + '_' + dt_string + '.csv')
        np.savetxt(grtName, YTest, delimiter=',')
        np.savetxt(prdName, YPred, delimiter=',')
