#!/usr/bin/env python

from builtins import zip
from builtins import object
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import logging
import warnings
import yaml

from allensdk.config.manifest import Manifest

from . import check_fi_shift
from . import sweep_functions as sf
from .step_analysis import StepAnalysis
from ipfx.sweep import SweepSet


class NoUsableSweepsException(Exception): pass


class FitStyle(object):
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


def _cap_check_indices(i):
    """Find the indices of the upward and downward pulses in the current trace `i`

    Assumes that there is a test pulse followed by the stimulus pulses (downward first)
    """
    di = np.diff(i)
    up_idx = np.flatnonzero(di > 0)
    down_idx = np.flatnonzero(di < 0)

    return up_idx[2::2].tolist(), down_idx[1::2].tolist()


def select_core1_trace(sweep_set, start, end,
        exceed_rheobase=39.0, min_isi_cv=0.3, fraction_isi_cv=0.2):
    """Identify a Core 1 long square sweep as the optimization target"""
    step_analysis = StepAnalysis(start, end)
    step_analysis.analyze(sweep_set)
    sweep_info = step_analysis.sweep_features()
    quality_values = [is_sweep_good_quality(sd["threshold_t"].values, start, end)
                      if sd.shape[0] > 0 else False
                      for sd in step_analysis.spikes_data()]
    sweep_info["quality"] = quality_values

    spiking_features = step_analysis.suprathreshold_sweep_features()
    spiking_amps = spiking_features[spiking_features["stim_amp"] > 0].sort_values("stim_amp")
    if spiking_amps.empty:
        raise RuntimeError("Cannot find rheobase sweep")
    rheobase_amp = spiking_amps["stim_amp"].values[0]
    unique_amps = np.unique(np.rint(spiking_amps["stim_amp"].values))

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

    if len(sweeps_to_use) == 0:
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
    """Determine target feature means and standard deviations

    Can use a minimum standard deviation to avoid over-weighting certain features
    that have extremely small variance.

    Parameters
    ----------
    sweep_data: DataFrame
        Rows have values for each sweep
    spike_data_list: list
        DataFrames of individual spike values for each sweep
    min_std_dict: dict (optional)
        If not supplied, values will be loaded from default YAML file

    Returns
    -------
    DataFrame of target feature means and standard deviations
    """

    if min_std_dict is None:
        # Load from default configuration file if not supplied
        with open(os.path.join(os.path.dirname(__file__), '../config/minimum_feature_standard_deviations.yaml')) as f:
            min_std_dict = yaml.load(f, Loader=yaml.Loader)

    # Sweep-level features
    exclude = ["stim_amp"]
    results = []
    sweep_cols = sweep_data.select_dtypes(include=[np.number]).columns.tolist()
    for col in sweep_cols:
        if col in exclude:
            continue
        mean = sweep_data[col].mean(skipna=True)
        std = sweep_data[col].std(skipna=True)
        if sweep_data.shape[0] == 1:
            std = 0
        if col in min_std_dict:
            std = max(std, min_std_dict[col])
        else:
            logging.debug("Sweep column {} did not have a minimum "
                "standard deviation specified".format(col))
        results.append({
            "name": col,
            "mean": mean,
            "stdev": std,
        })

    # Spike-level features
    all_cols = []
    for spike_data in spike_data_list:
        all_cols += spike_data.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = np.unique(all_cols)

    exclude_suffixes = ["_index", "_i", "_t"]
    for col in all_cols:
        if np.any([col.endswith(s) for s in exclude_suffixes]):
            continue
        sweep_means = []
        for sd in spike_data_list:
            if col not in sd:
                continue
            clip_mask = sd["clipped"].values
            sweep_means.append(sd.loc[~clip_mask, col].mean(skipna=True))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(sweep_means)
            std = np.nanstd(sweep_means)
        if col in min_std_dict:
            std = max(std, min_std_dict[col])
        else:
            logging.debug("Spike column {} did not have a minimum "
                "standard deviation specified".format(col))
        results.append({
            "name": col,
            "mean": mean,
            "stdev": std,
        })

    return DataFrame(results).set_index("name")


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
            if spike_data.shape[0] == 0:
                continue

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


def save_grand_averages(grand_up, grand_down, t, storage_directory):
    """Save to local storage to be loaded by NEURON fitting scripts"""
    Manifest.safe_mkdir(storage_directory)
    upfile = os.path.join(storage_directory, "upbase.dat")
    downfile = os.path.join(storage_directory, "downbase.dat")
    with open(upfile, 'w') as f:
        np.savetxt(f, np.column_stack((t, grand_up)))
    with open(downfile, 'w') as f:
        np.savetxt(f, np.column_stack((t, grand_down)))

    return upfile, downfile



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

    avg_grand_diff = Series(
        grand_diff[start_index:], index=t[start_index:]).rolling(rolling_average_length, min_periods=1).mean()
    escape_indexes = np.flatnonzero(np.abs(avg_grand_diff.values) > threshold) + start_index
    if len(escape_indexes) < 1:
        escape_index = len(t) - 1
    else:
        escape_index = escape_indexes[0]

    if escape_index == start_index:
        raise RuntimeError("The window for passive fitting was found to have zero duration")

    return t[escape_index]


def cap_check_grand_averages(sweep_set, extra_interval=2.0):
    """Average and baseline identical sections of capacitance check sweeps

    Parameters
    ----------
    sweep_set: SweepSet

    extra_interval: float (optional, default 2.0)
        Extra time prior to stimulus to include in average

    Returns
    -------
    grand_up: array
    grand_down: array
        Averages of responses to depolarizing (`grand_up`) and hyperpolarizing
        (`grand_down`) pulses
    t: array
        Time data for grand_up and grand_down in ms
    """
    grand_up_list = []
    grand_down_list = []
    for swp in sweep_set.sweeps:
        passive_delta_t = (swp.t[1] - swp.t[0]) * 1e3 # in ms
        extra_interval = 2. # ms
        extra = int(extra_interval / passive_delta_t)
        up_idxs, down_idxs = _cap_check_indices(swp.i)

        inter_stim_interval = up_idxs[0] - down_idxs[0]
        down_idxs.append(up_idxs[-1] + inter_stim_interval)

        avg_up_list = []
        avg_down_list = []
        for down_start, down_end, up_start, up_end in zip(
            down_idxs[:-1], up_idxs, up_idxs, down_idxs[1:]):
            avg_down = swp.v[(down_start - extra):down_end]
            avg_up = swp.v[(up_start - extra):up_end]
            avg_down_list.append(avg_down)
            avg_up_list.append(avg_up)
        avg_up = np.vstack(avg_up_list).mean(axis=0)
        avg_down = np.vstack(avg_down_list).mean(axis=0)

        grand_up_list.append(avg_up - avg_up[0:extra].mean())
        grand_down_list.append(avg_down - avg_down[0:extra].mean())

    grand_up = np.vstack(grand_up_list).mean(axis=0)
    grand_down = np.vstack(grand_down_list).mean(axis=0)
    t = passive_delta_t * np.arange(len(grand_up)) # in ms

    return grand_up, grand_down, t


def max_i_for_depol_block_check(*sweep_set_args):
    """Returns maximum current (in nA) to use for depolarization block check

    Parameters
    ----------
    sweep_set_args: SweepSet objects
        Current properties (sweep.i) values should be in pA

    Returns
    -------
    Maximum current used across all sweeps (in nA)
    """
    max_i = 0
    for sweep_set in sweep_set_args:
        if sweep_set is None:
            continue

        for swp in sweep_set.sweeps:
            max_i = max(np.max(swp.i), max_i)

    max_i += 10 # increase by 10 pA for additional safety
    max_i *= 1e-3 # convert from pA to nA

    return max_i


def swc_has_apical_compartments(swc_file):
    """ Returns whether SWC file has any apical dendrite compartments"""

    # Reading data into dataframe since we don't need a full, connected
    # Morphology object
    swc_data = pd.read_csv(swc_file, sep='\s+', comment='#', header=None)

    # SWC file conventions
    apical_code = 4
    compartment_type_column = 1

    return apical_code in swc_data[compartment_type_column].unique().tolist()

