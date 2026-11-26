#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Theory & Tools
# File:         literature.py
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Literature of the Theory & Tools pages (neural networks, time-series modelling and thermal
modelling): REFS {key: {type, label, cite, url}}, url(key), label(key).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import urllib.parse

#######################################################################################################################
# References
#######################################################################################################################
REFS = {'Incropera2007': {'type': 'Book', 'label': 'Incropera et al. 2007', 'cite': 'F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass Transfer, 6th ed., John Wiley & Sons, 2007.', 'url': None, 'topic': 'Heat conduction, convection and material properties', 'modes': [0]},
        'Wintrich2015': {'type': 'Book', 'label': 'Wintrich et al. 2015', 'cite': 'A. Wintrich, U. Nicolai, W. Tursky and T. Reimann, Application Manual Power Semiconductors, 2nd ed., SEMIKRON International, 2015.', 'url': None, 'topic': 'Losses, thermal Foster/Cauer networks and lifetime of power semiconductors', 'modes': [0]},
        'JESD51-14': {'type': 'Norm / standard / guideline', 'label': 'JEDEC JESD51-14', 'cite': 'JEDEC JESD51-14, Transient Dual Interface Test Method for the Measurement of the Thermal Resistance Junction-to-Case of Semiconductor Devices with Heat Flow Through a Single Path, JEDEC, 2010.', 'url': 'https://www.jedec.org', 'topic': 'Thermal impedance and Foster networks of semiconductors', 'modes': [0]},
        'Lee1995': {'type': 'Paper', 'label': 'Lee et al. 1995', 'cite': 'S. Lee, S. Song, V. Au and K. P. Moran, "Constriction/spreading resistance model for electronics packaging," Proc. ASME/JSME Thermal Engineering Joint Conference, vol. 4, pp. 199–206, 1995.', 'url': None, 'topic': 'Spreading resistance of a heated plate with convection (thermal stack designer)', 'modes': [0]},
        'Bierhoff2004': {'type': 'Paper', 'label': 'Bierhoff & Fuchs 2004', 'cite': 'F. Bierhoff and F. W. Fuchs, "Semiconductor losses in voltage source and current source IGBT converters based on analytical derivation," Proc. IEEE 35th Power Electronics Specialists Conference (PESC), pp. 2836–2842, 2004.', 'url': None, 'topic': 'Closed-form inverter losses with sinusoidal PWM', 'modes': [0]},
        'Graovac2009': {'type': 'Paper', 'label': 'Graovac & Pürschel 2009', 'cite': 'D. Graovac and M. Pürschel, "IGBT power losses calculation using the data-sheet parameters," Application Note, Infineon Technologies, 2009.', 'url': None, 'topic': 'Conduction and switching losses of IGBTs and diodes in inverters', 'modes': [0]},
        'Baliga2019': {'type': 'Book', 'label': 'Baliga 2019', 'cite': 'B. J. Baliga, Fundamentals of Power Semiconductor Devices, 2nd ed., Springer, 2019.', 'url': 'https://doi.org/10.1007/978-3-319-93988-9', 'topic': 'Si and SiC power devices, on-resistance and its temperature dependence', 'modes': [0]},
        'Lidow2019': {'type': 'Book', 'label': 'Lidow et al. 2019', 'cite': 'A. Lidow, M. de Rooij, J. Strydom, D. Reusch and J. Glaser, GaN Transistors for Efficient Power Conversion, 3rd ed., John Wiley & Sons, 2019.', 'url': None, 'topic': 'GaN HEMT characteristics, reverse conduction and switching losses', 'modes': [0]},
        'Dowell1966': {'type': 'Journal paper', 'label': 'Dowell 1966', 'cite': 'P. L. Dowell, Effects of eddy currents in transformer windings, Proceedings of the IEE, vol. 113, no. 8, pp. 1387-1394, 1966.', 'url': 'https://doi.org/10.1049/piee.1966.0236', 'topic': 'Skin and proximity losses of windings (layers of foil or round wire)', 'modes': [0]},
        'Kazimierczuk2014': {'type': 'Book', 'label': 'Kazimierczuk 2014', 'cite': 'M. K. Kazimierczuk, High-Frequency Magnetic Components, 2nd ed., John Wiley & Sons, 2014.', 'url': None, 'topic': 'Skin depth, AC resistance of round wires (Bessel solution), proximity effect', 'modes': [0]},
        'AQG324': {'type': 'Norm / standard / guideline', 'label': 'ECPE AQG 324', 'cite': 'ECPE AQG 324, Qualification of Power Modules for Use in Power Electronics Converter Units in Motor Vehicles, ECPE.', 'url': 'https://www.ecpe.org', 'topic': 'Power cycling and qualification of power modules', 'modes': [0, 1, 5]},
        'Goodfellow2016': {'type': 'Book', 'label': 'Goodfellow et al. 2016', 'cite': 'I. Goodfellow, Y. Bengio and A. Courville, Deep Learning, MIT Press, 2016.', 'url': 'https://www.deeplearningbook.org', 'topic': 'Neural networks, training, convolutional and recurrent networks', 'modes': [0]},
        'Hochreiter1997': {'type': 'Journal paper', 'label': 'Hochreiter & Schmidhuber 1997', 'cite': 'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.', 'url': 'https://doi.org/10.1162/neco.1997.9.8.1735', 'topic': 'LSTM memory cell and gates', 'modes': [0]},
        'Bai2018': {'type': 'Paper', 'label': 'Bai et al. 2018', 'cite': 'S. Bai, J. Z. Kolter and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.', 'url': 'https://arxiv.org/abs/1803.01271', 'topic': 'Dilated causal convolutions (TCN) vs. recurrent networks', 'modes': [0]},
        'Vaswani2017': {'type': 'Paper', 'label': 'Vaswani et al. 2017', 'cite': 'A. Vaswani et al., "Attention is all you need," Advances in Neural Information Processing Systems (NeurIPS), 2017.', 'url': 'https://arxiv.org/abs/1706.03762', 'topic': 'Transformer and self-attention', 'modes': [0]},
        'Raissi2019': {'type': 'Journal paper', 'label': 'Raissi et al. 2019', 'cite': 'M. Raissi, P. Perdikaris and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational Physics, vol. 378, pp. 686-707, 2019.', 'url': 'https://doi.org/10.1016/j.jcp.2018.10.045', 'topic': 'Physics-informed neural networks (PINN)', 'modes': [0]},
        'Cai2021': {'type': 'Journal paper', 'label': 'Cai et al. 2021', 'cite': 'S. Cai, Z. Wang, S. Wang, P. Perdikaris and G. E. Karniadakis, "Physics-informed neural networks for heat transfer problems," Journal of Heat Transfer, vol. 143, no. 6, 060801, 2021.', 'url': 'https://doi.org/10.1115/1.4050542', 'topic': 'PINNs for heat transfer: unknown boundary conditions, sparse sensors, power electronics', 'modes': [0]},
        'Lu2021': {'type': 'Journal paper', 'label': 'Lu et al. 2021', 'cite': 'L. Lu, P. Jin, G. Pang, Z. Zhang and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," Nature Machine Intelligence, vol. 3, pp. 218-229, 2021.', 'url': 'https://doi.org/10.1038/s42256-021-00302-5', 'topic': 'Operator networks with branch and trunk (DeepONet)', 'modes': [0]},
        'Wang2021': {'type': 'Journal paper', 'label': 'Wang et al. 2021', 'cite': 'S. Wang, H. Wang and P. Perdikaris, "Learning the solution operator of parametric partial differential equations with physics-informed DeepONets," Science Advances, vol. 7, no. 40, eabi8605, 2021.', 'url': 'https://doi.org/10.1126/sciadv.abi8605', 'topic': 'Physics-informed operator networks', 'modes': [0]},
        'Benner2015': {'type': 'Journal paper', 'label': 'Benner et al. 2015', 'cite': 'P. Benner, S. Gugercin and K. Willcox, "A survey of projection-based model reduction methods for parametric dynamical systems," SIAM Review, vol. 57, no. 4, pp. 483-531, 2015.', 'url': 'https://doi.org/10.1137/130932715', 'topic': 'Parametric model order reduction (pMOR), affine parameter dependence', 'modes': [0]},
        'Odabasioglu1998': {'type': 'Journal paper', 'label': 'Odabasioglu et al. 1998', 'cite': 'A. Odabasioglu, M. Celik and L. T. Pileggi, "PRIMA: Passive reduced-order interconnect macromodeling algorithm," IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 17, no. 8, pp. 645-654, 1998.', 'url': 'https://doi.org/10.1109/43.712097', 'topic': 'Block Krylov moment matching of RC networks (C, G, B)', 'modes': [0]},
        'Antoulas2005': {'type': 'Book', 'label': 'Antoulas 2005', 'cite': 'A. C. Antoulas, Approximation of Large-Scale Dynamical Systems, SIAM, 2005.', 'url': 'https://doi.org/10.1137/1.9780898718713', 'topic': 'Model order reduction: balanced truncation and Krylov moment matching', 'modes': [0]},
        'Holmes2012': {'type': 'Book', 'label': 'Holmes et al. 2012', 'cite': 'P. Holmes, J. L. Lumley, G. Berkooz and C. W. Rowley, Turbulence, Coherent Structures, Dynamical Systems and Symmetry, 2nd ed. Cambridge University Press, 2012.', 'url': 'https://doi.org/10.1017/CBO9780511919701', 'topic': 'Proper orthogonal decomposition (POD), method of snapshots, Galerkin projection', 'modes': [0]},
        'Ohlberger2016': {'type': 'Conference paper', 'label': 'Ohlberger and Rave 2016', 'cite': 'M. Ohlberger and S. Rave, "Reduced basis methods: Success, limitations and future challenges," Proceedings of the Conference Algoritmy, pp. 1-12, 2016.', 'url': 'https://arxiv.org/abs/1511.02021', 'topic': 'Kolmogorov n-width: why linear reduced bases fail for moving sources and transport', 'modes': [0]},
        'Kirchgaessner2023': {'type': 'Journal paper', 'label': 'Kirchgässner et al. 2023', 'cite': 'W. Kirchgässner, O. Wallscheid and J. Böcker, "Thermal neural networks: Lumped-parameter thermal modeling with state-space machine learning," Engineering Applications of Artificial Intelligence, vol. 117, 105537, 2023.', 'url': None, 'topic': 'Neural networks with thermal network structure for electric machines', 'modes': [0]}}



REFS.update({
    'Schirmer2024': {'type': 'Journal paper', 'label': 'Schirmer & Mporas 2024', 'cite': 'P. A. Schirmer and I. Mporas, "PyDTS: A Python toolkit for deep learning time series modelling," Entropy, vol. 26, no. 4, 311, 2024.', 'url': 'https://doi.org/10.3390/e26040311', 'topic': 'The PyDTS toolkit: data handling, models, tutorials and benchmarks', 'modes': [0, 1, 2]},
    'Kirchgaessner2021': {'type': 'Journal paper', 'label': 'Kirchgässner et al. 2021a', 'cite': 'W. Kirchgässner, O. Wallscheid and J. Böcker, "Estimating electric motor temperatures with deep residual machine learning," IEEE Transactions on Power Electronics, vol. 36, no. 7, pp. 7480-7488, 2021.', 'url': 'https://doi.org/10.1109/TPEL.2020.3045596', 'topic': 'CNN/RNN temperature estimation of a PMSM (dataset of Tutorial 3)', 'modes': [0]},
    'Kirchgaessner2021b': {'type': 'Journal paper', 'label': 'Kirchgässner et al. 2021b', 'cite': 'W. Kirchgässner, O. Wallscheid and J. Böcker, "Data-driven permanent magnet temperature estimation in synchronous motors with supervised machine learning: A benchmark," IEEE Transactions on Energy Conversion, vol. 36, no. 3, pp. 2059-2067, 2021.', 'url': 'https://doi.org/10.1109/TEC.2021.3052546', 'topic': 'Benchmark of ML models and EWMA features for magnet temperatures', 'modes': [0]},
    'Kingma2015': {'type': 'Conference paper', 'label': 'Kingma & Ba 2015', 'cite': 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," International Conference on Learning Representations (ICLR), 2015.', 'url': 'https://arxiv.org/abs/1412.6980', 'topic': 'Adam optimiser', 'modes': [0]},
    'Ljung1999': {'type': 'Book', 'label': 'Ljung 1999', 'cite': 'L. Ljung, System Identification: Theory for the User, 2nd ed., Prentice Hall, 1999.', 'url': None, 'topic': 'State-space and transfer-function identification (SP solver)', 'modes': [0]},
    'Hyndman2021': {'type': 'Book', 'label': 'Hyndman & Athanasopoulos 2021', 'cite': 'R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., OTexts, 2021.', 'url': 'https://otexts.com/fpp3/', 'topic': 'Time-series forecasting, cross validation of time series, error measures', 'modes': [0]},
})


#######################################################################################################################
# Functions
#######################################################################################################################
def url(key):
    """Link of a reference: DOI or issuer if known, else a Google Scholar search."""
    r = REFS[key]
    if r.get('url'):
        return r['url']
    return 'https://scholar.google.com/scholar?q=' + urllib.parse.quote(r['cite'])


def label(key):
    return REFS[key]['label']
