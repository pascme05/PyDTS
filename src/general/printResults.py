#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         printResults
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

#######################################################################################################################
# Function
#######################################################################################################################
def printResults(resultsApp, resultsAvg, setupDat):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("INFO: Printing Results on Console")

    ####################################################################################################################
    # Calculations
    ####################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print('\t|          |    FINITE STATES   |                           ESTIMATION                             |   PERCENT OF TOTAL  |')
    print('\t| item ID  | ACCURACY | F-SCORE |  R2-Score  |    TECA    |    RMSE    |     MAE     |     MAX     |    EST    |  TRUTH  |')
    print('\t|----------|----------|---------|------------|------------|------------|-------------|-------------|-----------|---------|')
    for i in range(0, len(setupDat['out'])):
        print('\t| %-8s |  %6.2f%% | %6.2f%% |  %6.2f%%   |  %6.2f%%   |  %8.2f  |  %8.2f   |  %8.2f   |  %6.2f%%  | %6.2f%% |' % (
            setupDat['out'][i], resultsApp[i, 0] * 100, resultsApp[i, 1] * 100, resultsApp[i, 2] * 100, resultsApp[i, 3]*100,
            resultsApp[i, 4], resultsApp[i, 5], resultsApp[i, 6], resultsApp[i, 7] * 100, resultsApp[i, 8] * 100))
        print('\t|----------|----------|---------|------------|------------|------------|-------------|-------------|-----------|---------|')
    print('\t|----------|----------|---------|------------|------------|------------|-------------|-------------|-----------|---------|')
    print('\t|    AVG   |  %6.2f%% | %6.2f%% |  %6.2f%%   |  %6.2f%%   |  %8.2f  |  %8.2f   |  %8.2f   |  %6.2f%%  | %6.2f%% |' % (
        resultsAvg[0] * 100, resultsAvg[1] * 100, resultsAvg[2] * 100, resultsAvg[3]*100, resultsAvg[4], resultsAvg[5], resultsAvg[6],
        resultsAvg[7] * 100, resultsAvg[8] * 100))
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------------------')