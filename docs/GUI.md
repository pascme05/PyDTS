# PyDTS GUI — User Guide

The PyDTS GUI is a local web application (Streamlit) around the unchanged toolkit: all parameters of `start.py` and
`mdlPara.py` are edited in the browser and stored as JSON, PyDTS runs in the same process exactly as from
`start.py`, and its predictions are shown as interactive charts. The GUI runs on your own machine only
(`http://localhost:8501`); no data leave it.

## 1. Installation and start

The GUI needs Streamlit, Plotly and graphviz in the PyDTS environment (numpy stays pinned to the toolkit's version):

```
venv\Scripts\python -m pip install -r gui/requirements.txt
venv\Scripts\python start_gui.py                      # opens http://localhost:8501
```

Stop the GUI with `Ctrl+C`. Server settings (port, localhost only, theme) are in `.streamlit/config.toml`.
The first run of a session takes a few seconds longer: PyTorch is imported then. With *Device* = 1 the networks
train on the first CUDA GPU (install the CUDA build of PyTorch, see the README).

## 2. Layout

| Area | Content |
|---|---|
| **Sidebar** | Focus mode, expert mode, configuration manager (load / save / download / import / reset), runs of this session |
| **Project › Setup** | Run settings (name, training, testing, plots, saving), experiment parameters, the operating modes |
| **Project › Input data** | Every dataset of `data/`: overview, signals, split into training/validation/test data, correlation |
| **Operating modes** | Simulation, Hyperparameter optimisation, Grid optimisation — each with the views *Workflow*, *Parameters*, *Inputs*, *Theory*, *Run*, *Results* |
| **Theory & Tools** | *Temperature prediction* (window and EWMA design calculator, window vs features, thermal background) and *Neural networks* (DNN, CNN, LSTM, attention, PINN, data and training on thermal examples) |
| **Results browser** | All GUI runs in `results/gui/` |
| **Help** | Tutorials, this guide and the parameter reference |

**Focus mode** (on by default) shows on the result and theory pages the answer, the key figures, the main chart and
the interpretation; switch it off to see footnotes, secondary tables and chart explanations. **Expert mode** shows
all parameters, including those that have no effect with the current settings (marked *inactive*).

## 3. Configurations

PyDTS has no config workbook: its parameters are set in `start.py` (`setupExp`, `setupDat`, `setupPar`) and
`mdlPara.py` (`setupMdl`). `python gui/build_config.py` reads both scripts and writes

- `config/default.xlsx` — the parameter table (sheets Exp, Dat, Par, Mdl with Name | Category | Description | Variable |
  Value | Unit), the source of the forms,
- `gui/schema.json` and `config/default.json` — schema and default configuration of the GUI,
- `config/<script>.json` — ready-to-load configurations of every setup in `setup/tutorial`, `setup/lecture` and
  `setup/journal` (e.g. *Tutorial_3* = motor temperatures).

Run it again when parameters are added to `start.py` or `mdlPara.py`. Lists (datasets, inputs, outputs, EWMA spans) are
entered as comma separated text. *Saved configs → Load* reads `config/<name>.json`, *Save* writes it.

## 4. Running a simulation

*Training* = 0 loads the stored model `mdl/mdl_<model>_<name>/model.pt`; *Training* = 1 trains it (continues from stored
weights if the folder exists). The *Run* view streams the console of PyDTS. Every run is stored in
`results/gui/<name>_<time>/` (`gui_run.pkl`, `gui_run.json`) and can be re-opened in the Results view or the browser.

**Data split.** Training, validation and test data never overlap. The validation data (early stopping, learning-rate
reduction) are the last share *rV* of the training data; with the ID split they are the last share *rV* of each
validation ID (or of every training ID if the validation IDs are empty or test IDs), so every non-test ID is trained on. The normalisation values come from
training + validation data, never from the test data. With shuffling, one fixed random order (seed 42) is used, so
inputs and outputs stay paired and training and testing see disjoint samples. The stored model is the epoch with the
lowest validation loss.

The **Results** view answers in a fixed order: the result in brief (answer, key figures, what it means), then one
tab per question — *Accuracy* (MAE, RMSE, MAX, R² per output as computed by `src/general/performance.py`), *Time
series*, *Error*, *Scatter*, *Training* (training log) and *Features* (feature ranking). The matplotlib figures of
PyDTS are under *Figures & files*.

## 5. Theory

The *Theory* view of a mode explains the configured model, with explorers that start from the configuration:

| Tab | Content |
|---|---|
| Overview | The chain datasets → normalisation → windows → model → back-scaling → errors |
| Data & windows | Window span T_W = W / f_s against the memory τ of the output, estimated from the autocorrelation of the first training dataset |
| Network | The PyTorch network of `src/model/models.py` for the current window (layers, shapes, weights per training window) |
| Training | Loss, Adam, early stopping, learning-rate reduction; the stored training log of the model |
| Errors | `performance()` of PyDTS on an adjustable example (bias, lag, noise, gain) |

The **Theory & Tools** pages are self-contained: *Temperature prediction › Design calculator* turns thermal time
constants and the sampling frequency into a window length and EWMA spans and can write them into the configuration.

## 6. Tutorials

Help › Tutorials loads a tutorial configuration (based on `setup/tutorial/*.py`) and opens its mode. Tutorial 3
(motor temperatures, DNN) and its CNN variant include the results of real runs to compare with.

## 7. Troubleshooting

| Problem | Remedy |
|---|---|
| Port 8501 in use | Stop the other instance or change `server.port` in `.streamlit/config.toml` |
| Changes in the code are not visible | Restart `start_gui.py` |
| "Could not load weights" / model not found | *Training* = 0 needs `mdl/mdl_<model>_<name>`; train once or use the name of a stored model |
| A config from an older version misses parameters | Missing parameters are filled with the defaults on loading |
| Training runs on the CPU although Device = 1 | The CPU build of PyTorch is installed — install the CUDA build (README) |
| "Stored model … was built for other settings" | Window, inputs or outputs changed since training: retrain or change the name |
