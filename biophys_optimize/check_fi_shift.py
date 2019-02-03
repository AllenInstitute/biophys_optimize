#!/usr/bin/env python

import numpy as np
from collections import Counter
import argparse

from allensdk.core.nwb_data_set import NwbDataSet
from biophys_optimize import sweep_functions as sf


def calculate_fi_curves(sweeps_input, data_set):
    c1_sweeps = sweeps_input["core_1_long_squares"]
    c1_ext = sf.sweep_set_extractor_from_list(c1_sweeps, data_set, sf.C1LS_START, sf.C1LS_END)
    c1_ext.process_spikes()
    coarse_fi_curve = zip(c1_ext.sweep_features("stim_amp"), c1_ext.sweep_features("avg_rate"))

    c2_sweeps = sweeps_input["core_2_long_squares"]

    # Use the Core 1 time to get a comparable F-I curve (i.e. look at first half
    # of the Core 2 sweep)
    c2_ext = sf.sweep_set_extractor_from_list(c2_sweeps, data_set, sf.C1LS_START, sf.C1LS_END)
    core2_amp_counter = Counter(c2_ext.sweep_features("stim_amp"))
    common_amps = [a[0] for a in core2_amp_counter.most_common(3)]

    core2_fi_curve = []
    core2_half_fi_curve = []
    for swp in c2_ext.sweeps():
        amp = swp.sweep_feature("stim_amp")
        if amp not in common_amps:
            continue
        swp.process_spikes()
        core2_half_fi_curve.append((amp, swp.sweep_feature("avg_rate")))

    return { "coarse": coarse_fi_curve, "core2_half": core2_half_fi_curve }


def estimate_fi_shift(sweeps_input, data_set):
    curve_data = calculate_fi_curves(sweeps_input, data_set)

    # Linear fit to original fI curve
    coarse_fi_sorted = sorted(curve_data["coarse"], key=lambda d: d[0])
    x = np.array([d[0] for d in coarse_fi_sorted], dtype=np.float64)
    y = np.array([d[1] for d in coarse_fi_sorted], dtype=np.float64)

    if np.all(y == 0): # original curve is all zero, so can't figure out shift
        return np.nan, 0

    last_zero_index = np.flatnonzero(y)[0] - 1
    A = np.vstack([x[last_zero_index:], np.ones(len(x[last_zero_index:]))]).T
    m, c = np.linalg.lstsq(A, y[last_zero_index:])[0]

    # Relative error of later traces to best-fit line
    if len(curve_data["core2_half"]) < 1:
        return np.nan, 0
    # FIX TO RECTIFY PREDICTED FI CURVE
    x_shift = [amp - (freq - c) / m for amp, freq in curve_data["core2_half"]]
    return np.mean(x_shift), len(x_shift)

