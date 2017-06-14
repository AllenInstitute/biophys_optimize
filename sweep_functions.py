#!/usr/bin/env python

import numpy as np
import lims_utils
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor, _step_stim_amp


C1LS_START = 1.02 # seconds
C1LS_END = 2.02 # seconds
C2LS_START = 1.02 # seconds
C2LS_END = 3.02 # seconds


def sweep_set_extractor_from_list(sweep_list, data_set, start, end, jxn=None):
    v_set = []
    t_set = []
    i_set = []
    for s in sweep_list:
        v, i, t = lims_utils.get_sweep_v_i_t_from_set(data_set, s)
        if jxn:
            v += jxn
        v_set.append(v)
        t_set.append(t)
        i_set.append(i)

    ext = EphysSweepSetFeatureExtractor(t_set, v_set, i_set, start=start,
                                        end=end)
    for swp in ext.sweeps():
        swp.set_stimulus_amplitude_calculator(_step_stim_amp)

    return ext


def slow_trough_norm_t(swp):
    threshold_t = swp.spike_feature("threshold_t")
    slow_trough_t = swp.spike_feature("slow_trough_t", include_clipped=True)
    trough_t = swp.spike_feature("trough_t", include_clipped=True)

    # if slow trough is undefined, use overall trough value instead
    nan_mask = np.isnan(slow_trough_t)
    slow_trough_t[nan_mask] = trough_t[nan_mask]

    if len(threshold_t) == 0:
        return np.array([])
    elif len(threshold_t) == 1:
        return np.array([(slow_trough_t[0] - threshold_t[0]) / (swp.end - threshold_t[0])])

    isis = np.diff(threshold_t)
    isis = np.append(isis, (swp.end - threshold_t[-1]))
    trough_intervals = slow_trough_t - threshold_t
    return trough_intervals / isis


def slow_trough_delta_voltage_feature(swp):
    return spike_feature_difference(swp, "fast_trough_v", "slow_trough_v")


def spike_feature_difference(swp, key_1, key_2, zero_nans=True):
    delta_values = (swp.spike_feature(key_1, include_clipped=True) -
                    swp.spike_feature(key_2, include_clipped=True))
    if zero_nans:
        delta_values[np.isnan(delta_values)] = 0.
    return delta_values