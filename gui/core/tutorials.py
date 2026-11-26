#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         tutorials.py
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
Tutorials of the GUI (page Help › Tutorials, shown when TUTORIALS is not empty): guided scenarios per operating mode.
Every tutorial is a config (config/tutorials/<id>.json) made of a base config and 'changes' {'Sec.cat.var': value}
(run settings: '<RUN_SECTION>.var'). Write the configs with 'python -m gui.core.tutorials'. The 'look' texts must
come from runs of the tutorial configs — never invent the numbers.

Entry: {'id', 'mode', 'level' (Beginner | Intermediate | Advanced), 'minutes', 'base' (config name, default
'default'), 'title', 'story', 'goals' [..], 'changes' {..}, 'steps' [..], 'look' (markdown), 'exercise' [..]}
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core.config import CONFIG_DIR, from_json, to_json, default_config, read_config

import copy
import os

TUT_DIR = os.path.join(CONFIG_DIR, 'tutorials')
RUN = {}                                                              # run settings of all tutorials (e.g. save off)

#######################################################################################################################
# Tutorials
#######################################################################################################################
TUTORIALS = [
    dict(id='t3_temperature', mode=0, level='Beginner', minutes=10, base='Tutorial_3', changes={},
         title="Motor temperatures with a DNN (Tutorial 3)",
         story="A permanent-magnet synchronous motor on a test bench: currents, voltages, speed, torque, ambient and "
               "coolant temperature are measured; the winding, tooth, yoke and magnet temperatures should be "
               "predicted without sensors (Kirchgässner et al., ID based split).",
         goals=["load a stored model and test it", "read MAE, RMSE, R² and the largest error per output",
                "relate the errors to the window length and the thermal memory"],
         steps=["Workflow: the chain data → windows (W = 1000 samples) → DNN → errors; Training = 0 loads "
                "mdl/mdl_DNN_Tutorial_3.",
                "Inputs › Signals: the stator temperatures follow the currents within minutes, the magnet much "
                "slower.",
                "Theory › Data & windows: compare the window span with the memory τ of each output.",
                "Run (20 – 45 s on a CPU, a few seconds with Device = GPU), then Results."],
         look="With the stored DNN: R² = 0.90 and a mean absolute error of 2.2 K averaged over the four "
              "temperatures; the largest single error is 17.4 K, on the magnet (T_rm). The magnet temperature "
              "T_rm is the hardest output (R² = 0.63) — it is far from the measured stator quantities and has the "
              "longest thermal memory.",
         exercise=["Switch Parameters › Model to CNN and run again (tutorial 'CNN vs DNN').",
                   "Look at Results › Scatter for T_rm: is the hot end under- or overestimated?"]),
    dict(id='t3_cnn', mode=0, level='Intermediate', minutes=10, base='Tutorial_3', changes={'Par.model': 'CNN'},
         title="CNN vs DNN for motor temperatures",
         story="The same motor data as Tutorial 3, now with the convolutional network of PyDTS (stored model "
               "mdl/mdl_CNN_Tutorial_3); the PyDTS paper found the CNN to be the best model for this task.",
         goals=["compare two architectures on the same split", "read the network size in Theory › Network"],
         steps=["Theory › Network: the CNN shares its filters over time; compare its weights with the DNN.",
                "Run and compare the key figures with the DNN tutorial."],
         look="With the stored CNN: mean absolute error 1.7 K (DNN: 2.2 K), RMSE 2.0 K (DNN: 2.8 K) and R² = "
              "0.93 (DNN: 0.90) — the CNN is better on every output, most on the magnet (R² 0.73 vs 0.63), whose "
              "slow heating the shared filters capture better. Compare the outputs one by one in Results › "
              "Accuracy; the Results browser keeps both runs.",
         exercise=["Which output gains most from the CNN, and why (fast vs slow thermal paths)?"]),
    dict(id='t1_denoise', mode=0, level='Beginner', minutes=5, base='Tutorial_1', changes={},
         title="Denoising / energy disaggregation (Tutorial 1)",
         story="The total power of a house (AMPds2) is split into the consumption of single appliances with a CNN.",
         goals=["read multi-output errors", "see the on/off accuracy (ACC, F1)"],
         steps=["Inputs › Signals: total power and appliance powers.", "Run, then Results › Time series."],
         look="Results › Accuracy lists MAE, F1 and ACC per appliance — compare appliances with clear on/off cycles "
              "with small, continuous loads.",
         exercise=["Change the window length and compare the errors."]),
    dict(id='t2_forecast', mode=0, level='Beginner', minutes=5, base='Tutorial_2', changes={},
         title="Load forecasting (Tutorial 2)",
         story="Electric power consumption of a city is predicted from its own past with a DNN on a one-day window.",
         goals=["understand the window of a forecast"],
         steps=["Theory › Data & windows: one window spans 144 samples (one day of 10-min data).", "Run."],
         look="Compare measured and predicted power in Results › Time series, especially at the daily peaks.",
         exercise=["Try the LSTM model."]),
    dict(id='t5_degradation', mode=0, level='Intermediate', minutes=5, base='Tutorial_5', changes={},
         title="Battery degradation (Tutorial 5)",
         story="Capacity of eVTOL battery cells over their cycles is predicted with an LSTM.",
         goals=["model slow degradation trends"], steps=["Run, then Results › Time series."],
         look="Check in Results › Time series how closely the slow capacity fade is followed.", exercise=[]),
    dict(id='t6_crossdomain', mode=0, level='Intermediate', minutes=5, base='Tutorial_6', changes={},
         title="Cross-domain modelling (Tutorial 6)",
         story="The DC current of an electric vehicle is predicted from driving and heating data with an LSTM.",
         goals=["model a vehicle quantity from other domains"], steps=["Run, then Results."],
         look="Results › Error shows when the largest deviations occur.", exercise=[]),
]


#######################################################################################################################
# Functions
#######################################################################################################################
def merge_changes(cfg, t):
    """Config of a tutorial: base config + run settings + changes 'Section.cat.var' (or '<RUN_SECTION>.var')."""
    cfg = copy.deepcopy(cfg)
    cfg[A.RUN_SECTION].update(RUN)
    for k, v in t['changes'].items():
        *path, last = k.split('.')
        d = cfg
        for p in path:
            d = d.setdefault(p, {})
        d[last] = copy.deepcopy(v)
    if A.MODE_VAR:
        cfg[A.RUN_SECTION][A.MODE_VAR] = t['mode']
    return cfg


def base(t):
    return default_config() if t.get('base', 'default') == 'default' else read_config(t['base'])


def config_file(tid):
    return os.path.join(TUT_DIR, f"{tid}.json")


def get(tid):
    return next(t for t in TUTORIALS if t['id'] == tid)


def load(tid):
    """Config of a tutorial (config/tutorials/<id>.json, generated from base and changes if the file is missing)."""
    f = config_file(tid)
    if os.path.isfile(f):
        with open(f, 'r', encoding='utf-8') as fh:
            return from_json(fh.read())
    return merge_changes(base(get(tid)), get(tid))


def write_all():
    """Writes config/tutorials/<id>.json for all tutorials."""
    os.makedirs(TUT_DIR, exist_ok=True)
    for t in TUTORIALS:
        with open(config_file(t['id']), 'w', encoding='utf-8') as fh:
            fh.write(to_json(merge_changes(base(t), t)))


if __name__ == '__main__':
    write_all()
