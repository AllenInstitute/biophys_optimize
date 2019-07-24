#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import re
import os
import sys
import logging
import yaml

from allensdk.config.manifest import Manifest

import .check_fi_shift as check_fi_shift
import .sweep_functions as sf


DEFAULT_SEEDS = [1234, 1001, 4321, 1024, 2048]


class NoUsableSweepsException(Exception): pass


class FitStyle():
    F6 = "f6"
    F6_NOAPIC = "f6_noapic"

    F9 = "f9"
    F9_NOAPIC = "f9_noapic"

    F12 = "f12"
    F12_NOAPIC = "f12_noapic"

    F13 = "f13"
    F13_NOAPIC = "f13_noapic"

    FIT_STAGE_MAP = { F6: F9,
                      F6_NOAPIC: F9_NOAPIC,
                      F12: F13,
                      F12_NOAPIC: F13_NOAPIC }

    FIT_NOAPIC_MAP = { F6: F6_NOAPIC,
                       F9: F9_NOAPIC,
                       F12: F12_NOAPIC,
                       F13: F13_NOAPIC }

    @staticmethod
    def get_fit_types(has_apical, is_spiny, width):
        if is_spiny:
            if width < 0.8:
                fit_types = [ FitStyle.F6, FitStyle.F12 ]
            else:
                fit_types = [ FitStyle.F6 ]
        else:
            if width > 0.8:
                fit_types = [ FitStyle.F6, FitStyle.F12 ]
            else:
                fit_types = [ FitStyle.F12 ]

        if not has_apical:
            fit_types = [ FitStyle.FIT_NOAPIC_MAP[fit_type] for fit_type in fit_types]

        return fit_types

    @staticmethod
    def map_stage_2(stage_1):
        return  FitStyle.FIT_STAGE_MAP[stage_1]


def get_cap_check_indices(i):
    """Find the indices of the upward and downward pulses in the current trace `i`

    Assumes that there is a test pulse followed by the stimulus pulses (downward first)
    """
    di = np.diff(i)
    up_idx = np.flatnonzero(di > 0)
    down_idx = np.flatnonzero(di < 0)

    return up_idx[2::2], down_idx[1::2]


def select_core1_trace(sweep_set, start, end,
        exceed_rheobase=39.0, min_isi_cv=0.3, fraction_isi_cv=0.2):
    """Identify a Core 1 long square sweep as the optimization target"""
    step_analysis = StepAnalysis(start, end)
    step_analysis.analyze(sweep_set)
    sweep_info = step_analysis.sweep_features()
    quality_values = [is_sweep_good_quality(sd["threshold_t"].values, start, end)
                      for sd in step_analysis.spikes_data()]
    sweep_info["quality"] = quality_values

    spiking_features = step_analysis.suprathreshold_sweep_features()
    spiking_amps = spiking_features[spiking_features["stim_amp"] > 0].sort_values("stim_amp")
    if spiking_amps.empty:
        raise RuntimeError("Cannot find rheobase sweep")
    rheobase_amp = spiking_amps.iloc[0]
    unique_amps = np.unique(np.rint(spiking_amps.values))

    sweep_to_use_mean_isi_cv = 1e12
    sweeps_to_use = []
    for a in unique_amps:
        if a < exceed_rheobase + rheobase_amp:
            continue
        mask = np.rint(sweep_info["stim_amp"].values) == a
        quality = sweep_info.loc[mask, "quality"].values
        if np.sum(quality) == 0:
            continue
        isi_cv = sweep_info.loc[mask, "isi_cv"].values[quality]
        if ((sweep_to_use_mean_isi_cv >= min_isi_cv) and
            np.all(isi_cv < (1. - fraction_isi_cv) * sweep_to_use_mean_isi_cv)):
            sweeps_to_use = np.array(sweep_set.sweeps)[mask][quality].tolist()
            sweep_to_use_mean_isi_cv = isi_cv.mean()

    if len(sweep_to_use) == 0:
        logging.info("Could not find appropriate core 1 sweep!")
        return []
    else:
        return sweeps_to_use


def is_sweep_good_quality(spike_times, start, end, min_rate=5):
    """Check whether the sweep passes several quality criteria:
    - Matches or exceeds defined minimum rate
    - Is not transient, defined as the time from the last spike to the
      end of the stimulus being twice as long as the average of the last two
      interspike intervals
    - Does not have a pause (defined by has_pause())

    Parameters
    ----------
    spike_times: array

    end: float
        End time of stimulus interval

    min_rate: float
        Minimum firing rate to consider (spikes/sec)

    Returns
    -------
    good_quality: bool
        Whether sweep should be considered for fitting
    """
    rate = np.sum((spike_times > start) & (spike_times < end)) / (end - start)

    if rate < min_rate:
        return False

    if len(spike_times) < 3:
        logging.debug("Need at least three spikes to determine sweep quality")
        return False

    time_to_end = end - spike_times[-1]
    avg_end_isi = ((spike_times[-1] - spike_times[-2]) + (spike_times[-2] - spike_times[-3])) / 2.

    if time_to_end > 2. * avg_end_isi:
        return False

    isis = np.diff(spike_times)
    if has_pause(isis):
        return False

    return True


def has_pause(isis):
    """Check whether a set of ISIs has a pause

    A pause here is an interspike interval (ISI) that is at least three times
    longer than its adjacent ISIs.

    Parameters
    ----------
    isis: array

    Returns
    -------
    Whether set of ISIs contains a pause
    """
    if len(isis) <= 2:
        return False

    for i, isi in enumerate(isis[1:-1]):
        if isi > 3 * isis[i + 1 - 1] and isi > 3 * isis[i + 1 + 1]:
            return True
    return False


def target_features(sweep_data, spike_data_list, min_std_dict=None):
    """Determine target features from sweeps in set extractor"""
    ext.process_spikes()
    for swp in ext.sweeps():
        swp.sweep_feature("v_baseline") # Pre-compute all the baselines, too
        swp.process_new_spike_feature("slow_trough_norm_t",
                                      sf.slow_trough_norm_t,
                                      affected_by_clipping=True)
        swp.process_new_spike_feature("slow_trough_delta_v",
                                      sf.slow_trough_delta_voltage_feature,
                                      affected_by_clipping=True)


    sweep_keys = swp.sweep_feature_keys()
    spike_keys = swp.spike_feature_keys()

    if min_std_dict is None:
        # Load from default configuration file if not supplied
        with open(os.path.join(os.path.dirname(__file__), '../config/minimum_feature_standard_deviations.yaml')) as f:
            min_std_dict = yaml.load(f)

    target_features = []
    for k in min_std_dict:
        if k in sweep_keys:
            values = ext.sweep_features(k)
        elif k in spike_keys:
            values = ext.spike_feature_averages(k)
        else:
            logging.debug("Could not find feature %s", k)
            continue

        t = {"name": k, "mean": float(values.mean()), "stdev": float(values.std())}
        if min_std_dict[k] > t["stdev"]:
            t["stdev"] = min_std_dict[k]
        target_features.append(t)

    return target_features


def select_core_1_or_core_2_sweeps(
        core_1_lsq, core_1_start, core_1_end,
        core_2_lsq, core_2_start, core_2_end,
        fi_shift_threshold=30.0):
    """Identify the sweep or sweeps to use as targets for optimization

    Prefers Core 2 sweeps because they are longer and usually have repeats.
    Selects a Core 1 sweep if the Core 2 sweeps do not exist or if the
    f-I curve has shifted by at least 30 pA (since Core 1 occurs earlier in the
    overall experimental protocol)

    Parameters
    ----------
    core_1_lsq: SweepSet
        "Core 1" long-square sweeps (1 second long stimulus, no repeats expected)
    core_1_start: float
        Start time of stimulus interval for Core 1 sweeps
    core_1_end: float
        End time of stimulus interval for Core 1 sweeps
    core_2_lsq: SweepSet
        "Core 2" long-square sweeps (2 seconds long stimulus, repeats expected)
    core_2_start: float
        Start time of stimulus interval for Core 2 sweeps
    core_2_end: float
        End time of stimulus interval for Core 2 sweeps
    fi_shift_threshold: float (optional, default 30.0)
        Maximum allowed f-I curve shift to still select Core 2 sweeps

    Returns
    -------
    sweeps_to_fit: SweepSet

    start: float
        time of stimulus start (in seconds)
    end: float
        time of stimulus end (in seconds)
    """

    use_core_1 = False
    if len(core_2_lsq.sweeps) == 0:
        logging.info("No Core 2 sweeps available")
        use_core_1 = True
    else:
        fi_shift = check_fi_shift.estimate_fi_shift(
            core_1_lsq, core_1_start, core_1_end,
            core_2_lsq, core_2_start, core_2_end)
        if abs(fi_shift) > fi_shift_threshold:
            logging.info(
                "f-I curve shifted by {} (exceeding +/- {}); "
                "using Core 1".format(fi_shift, fi_shift_threshold))
            use_core_1 = True

    if not use_core_1:
        # Try to find Core 2 sweeps that work
        core2_analysis = StepAnalysis(core_2_start, core_2_end)
        core2_analysis.analyze(core_2_lsq)
        core2_sweep_features = core2_analysis.sweep_features()
        stim_amps = np.rint(core2_sweep_features["stim_amp"].values)

        n_good = {}
        sweeps_by_amp = {}
        for amp, swp, spike_data in zip(stim_amps, core_2_lsq.sweeps,
                core2_analysis.spikes_data()):
            spike_times = spike_data["threshold_t"].values
            if is_sweep_good_quality(spike_times, core_2_start, core_2_end):
                if amp in n_good:
                    n_good[amp] += 1
                    sweeps_by_amp[amp].append(swp)
                else:
                    n_good[amp] = 1
                    sweeps_by_amp[amp] = [swp]

        if len(n_good) == 0 or max(n_good.values()) <= 1:
            logging.info("Not enough good Core 2 traces; using Core 1")
            use_core_1 = True

    if use_core_1:
        sweeps_to_fit = select_core1_trace(core_1_lsq, core_1_start, core_1_end)
        start = core_1_start
        end = core_1_end
    else:
        best_amp = max(n_good, key=(lambda key: n_good[key]))
        sweeps_to_fit = sweeps_by_amp[best_amp]
        start = core_2_start
        end = core_2_end

    return SweepSet(sweeps_to_fit), start, end


def prepare_for_passive_fit(sweeps, bridge_avg, is_spiny, data_set,
        storage_directory, threshold=0.2, start_time=4.0):
    """Collect information for passive fit variations on capacitance-check sweeps

    Parameters
    ----------
    sweeps : list
        list of sweep numbers of capacitance-check sweeps
    bridge_avg: float
        average of bridge-balance value during the sweeps
    is_spiny: bool
        True if neuron has dendritic spines
    data_set: NwbDataSet
        container of sweep data
    storage_directory: str
        path to storage directory
    threshold: float (optional, default 0.2)
        Fraction that up and down traces can diverge before the fitting window
        ends
    start_time: float (optional, default 4.0)
        Start time (in ms) of passive fitting window

    Returns
    -------
    paths: dict
        key-value set of relevant file paths
    passive_info: dict
        information about fitting for NEURON
    """
    if len(sweeps) == 0:
        logging.info("No cap check trace found")
        return {}, {"should_run": False}

    grand_up, grand_down, t = cap_check_grand_averages(sweeps, data_set)

    # Save to local storage to be loaded by NEURON fitting scripts
    Manifest.safe_mkdir(storage_directory)
    upfile = os.path.join(storage_directory, "upbase.dat")
    downfile = os.path.join(storage_directory, "downbase.dat")
    with open(upfile, 'w') as f:
        np.savetxt(f, np.column_stack((t, grand_up)))
    with open(downfile, 'w') as f:
        np.savetxt(f, np.column_stack((t, grand_down)))

    paths = {
        "up": upfile,
        "down": downfile,
    }

    escape_time = passive_fit_window(grand_up, grand_down, t, start_time, threshold)

    passive_info = {
        "should_run": True,
        "bridge": bridge_avg,
        "fit_window_start": start_time,
        "fit_window_end": escape_time,
        "electrode_cap": 1.0,
        "is_spiny": is_spiny,
    }

    return paths, passive_info


def passive_fit_window(grand_up, grand_down, t, start_time=4.0, threshold=0.2,
        rolling_average_length=100):
    """Determine how long the upward and downward responses are consistent

    Parameters
    ----------
    grand_up: array
        Average of positive-going capacitance check voltage responses
    grand_down: array
        Average of negative-going capacitance check voltage responses
    t: array
        Time points for grand_up and grand_down
    start_time: float (optional, default 4.0)
        Start time (in units of t) of passive fitting window
    threshold: float (optional, default 0.2)
        Fraction to which up and down traces can diverge before the fitting
        window ends
    rolling_average_length: int (optional, default 100)
        Length (in points) of window for rolling mean for divergence check

    Returns
    -------
    escape_time: float
        Time of divergence greater than `threshold`, or end of trace
    """
    # Determine for how long the upward and downward responses are consistent
    if not len(grand_up) == len(grand_down):
        raise ValueError("grand_up and grand_down must have the same length")
    if not len(grand_up) == len(t):
        raise ValueError("t and grand_up/grand_down must have the same length")

    grand_diff = (grand_up + grand_down) / grand_up
    start_index = np.flatnonzero(t >= start_time)[0]

    print("start_index", start_index)
    avg_grand_diff = Series(
        grand_diff[start_index:], index=t[start_index:]).rolling(rolling_average_length, min_periods=1).mean()
    print(avg_grand_diff.values)
    escape_indexes = np.flatnonzero(np.abs(avg_grand_diff.values) > threshold) + start_index
    if len(escape_indexes) < 1:
        escape_index = len(t) - 1
    else:
        escape_index = escape_indexes[0]

    if escape_index == start_index:
        raise RuntimeError("The window for passive fitting was found to have zero duration")

    return t[escape_index]


def cap_check_grand_averages(sweeps, data_set):
    """Average and baseline identical sections of capacitance check sweeps

    Parameters
    ----------
    sweeps: list
        list of sweep numbers of capacitance check sweeps
    data_set: NwbDataSet
        container of sweep data

    Returns
    -------
    grand_up: ndarray
    grand_down: ndarray
        Averages of responses to depolarizing (`grand_up`) and hyperpolarizing
        (`grand_down`) pulses
    t: ndarray
        Time data for grand_up and grand_down in ms
    """
    initialized = False
    for s in sweeps:
        v, i, t = sf.get_sweep_v_i_t_from_set(data_set, s)
        passive_delta_t = (t[1] - t[0]) * 1e3 # in ms
        extra_interval = 2. # ms
        extra = int(extra_interval / passive_delta_t)
        up_idxs, down_idxs = get_cap_check_indices(i)

        down_idx_interval = down_idxs[1] - down_idxs[0]
        inter_stim_interval = up_idxs[0] - down_idxs[0]
        for j in range(len(up_idxs)):
            if j == 0:
                avg_up = v[(up_idxs[j] - extra):down_idxs[j + 1]]
                avg_down = v[(down_idxs[j] - extra):up_idxs[j]]
            elif j == len(up_idxs) - 1:
                avg_up = avg_up + v[(up_idxs[j] - extra):up_idxs[j] + inter_stim_interval]
                avg_down = avg_down + v[(down_idxs[j] - extra):up_idxs[j]]
            else:
                avg_up = avg_up + v[(up_idxs[j] - extra):down_idxs[j + 1]]
                avg_down = avg_down + v[(down_idxs[j] - extra):up_idxs[j]]
        avg_up /= len(up_idxs)
        avg_down /= len(up_idxs)
        if not initialized:
            grand_up = avg_up - avg_up[0:extra].mean()
            grand_down = avg_down - avg_down[0:extra].mean()
            initialized = True
        else:
            grand_up = grand_up + (avg_up - avg_up[0:extra].mean())
            grand_down = grand_down + (avg_down - avg_down[0:extra].mean())
    grand_up /= len(sweeps)
    grand_down /= len(sweeps)
    t = passive_delta_t * np.arange(len(grand_up)) # in ms

    return grand_up, grand_down, t


def max_i_for_depol_block_check(sweeps_input, data_set):
    """Determine highest step to check for depolarization block"""
    noise_1_sweeps = sweeps_input["seed_1_noise"]
    noise_2_sweeps = sweeps_input["seed_2_noise"]
    step_sweeps = sweeps_input["core_1_long_squares"]
    all_sweeps = noise_1_sweeps + noise_2_sweeps + step_sweeps

    max_i = 0
    for s in all_sweeps:
        v, i, t = sf.get_sweep_v_i_t_from_set(data_set, s)
        if np.max(i) > max_i:
            max_i = np.max(i)
    max_i += 10 # add 10 pA
    max_i *= 1e-3 # convert to nA
    return max_i

def preprocess(data_set, swc_data, dendrite_type_tag,
               sweeps, bridge_avg, storage_directory,
               seeds=DEFAULT_SEEDS, jxn=-14.0):
    """ Perform preprocessing tasks prior to passive fitting and optimization

    Parameters
    ----------
    data_set: AibsDataSet
    swc_data: dataframe
    dendrite_type_tag: str
    sweeps: dict
    bridge_avg: float
    storage_directory: str
    seeds: list
    jxn: float (optional, default -14.0)

    Returns
    -------

    """

    # TODO - Change this from a long string check
    if dendrite_type_tag == "dendrite type - aspiny":
        is_spiny = False
        logging.debug("is not spiny")
    else:
        is_spiny = True
        logging.debug("is spiny or sparsely spiny")

    # Check for fi curve shift to decide to use core1 or core2

    paths, passive_info = prepare_for_passive_fit(sweeps["cap_checks"],
                                                  bridge_avg,
                                                  is_spiny, data_set,
                                                  storage_directory)

    ext = sf.sweep_set_extractor_from_list(sweeps_to_fit, data_set, start, end, jxn=jxn)
    targets = target_features(ext)
    max_i = max_i_for_depol_block_check(sweeps, data_set)

    # Decide which fit(s) we are doing
    width = [target["mean"] for target in targets if target["name"] == "width"][0]

    has_apical = False
    if 4 in pd.unique(swc_data[1]):
        has_apical = True
        logging.debug("Has apical dendrite")
    else:
        logging.debug("Does not have apical dendrite")


    fit_types = FitStyle.get_fit_types(has_apical=has_apical,
                                       is_spiny=is_spiny,
                                       width=width)

    stage_1_tasks = [{"fit_type": fit_type, "seed": seed} for seed in seeds
            for fit_type in fit_types]

    stage_2_tasks = [{"fit_type": FitStyle.map_stage_2(fit_type), "seed": seed} for seed in seeds
            for fit_type in fit_types]

    swp = ext.sweeps()[0]
    stim_amp = swp.sweep_feature("stim_amp")
    stim_dur = swp.end - swp.start

    v_baseline = [target["mean"] for target in targets if target["name"] == "v_baseline"][0]

    return paths, {
        "is_spiny": is_spiny,
        "has_apical": has_apical,
        "sweeps_to_fit": sweeps_to_fit.tolist(),
        "junction_potential": jxn,
        "max_stim_test_na": max_i,
        "v_baseline": v_baseline,
        "stimulus": {
            "amplitude": 1e-3 * stim_amp,
            "delay": 1e3,
            "duration": 1e3 * stim_dur,
        },
        "target_features": targets,
        "sweeps": sweeps,
    }, passive_info, stage_1_tasks, stage_2_tasks

