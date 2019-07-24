#!/usr/bin/env python

import numpy as np
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor, _step_stim_amp
from ipfx.sweep import Sweep, SweepSet
import ipfx.stim_features as stf


C1LS_START = 1.02 # seconds
C1LS_END = 2.02 # seconds
C2LS_START = 1.02 # seconds
C2LS_END = 3.02 # seconds


def sweeps_from_nwb(nwb_data, sweep_number_list):
    """ Generate a SweepSet object from an NWB reader and list of sweep numbers

    Sweeps should be in current-clamp mode.

    Parameters
    ----------
    nwb_data: NwbReader
    sweep_number_list: list
        List of sweep numbers

    Returns
    -------
    sweeps: SweepSet
    stim_start: float
        Start time of stimulus (seconds)
    stim_end: float
        End time of stimulus (seconds)
    """

    sweep_list = []
    start = None
    dur = None
    for sweep_number in sweep_number_list:
        sweep_data = nwb_data.get_sweep_data(sweep_number)
        sampling_rate = sweep_data["sampling_rate"]
        dt = 1.0 / sampling_rate
        t = np.arange(0, len(sweep_data["stimulus"])) * dt
        v = sweep_data["response"]
        i = sweep_data["stimulus"]
        sweep = Sweep(t=t,
                      v=v,
                      i=i,
                      sampling_rate=sampling_rate,
                      sweep_number=sweep_number,
                      clamp_mode="CurrentClamp",
                      epochs=None,
                      )
        sweep_list.append(sweep)
        start, dur, _, _, _ = stf.get_stim_characteristics(i, t)
    if start is None or dur is None:
        return SweepSet(sweep_list), None, None
    else:
        return SweepSet(sweep_list), start, start + dur


def get_sweep_v_i_t_from_set(data_set, sweep_number):
    sweep_data = data_set.get_sweep(sweep_number)
    i = sweep_data["stimulus"] # in A
    v = sweep_data["response"] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV
    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)
    return v, i, t


def sweep_set_extractor_from_list(sweep_list, data_set, start, end, jxn=None):
    v_set = []
    t_set = []
    i_set = []
    for s in sweep_list:
        v, i, t = get_sweep_v_i_t_from_set(data_set, s)
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
