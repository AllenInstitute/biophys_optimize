import numpy as np
from collections import Counter

from biophys_optimize import sweep_functions as sf
from .step_analysis import StepAnalysis


def fi_curve(sweep_set, start, end):
    """ Estimate the f-I curve from a set of sweeps

    Parameters
    ----------
    sweep_set: SweepSet
    start: float
        Start time of stimulus interval
    end: float
        End time of stimulus interval

    Returns
    -------
    amps: array
    rates: array
    """
    step_analysis = StepAnalysis(start, end)
    step_analysis.analyze(sweep_set)
    sweep_features = step_analysis.sweep_features()

    amps = sweep_features["stim_amp"].values
    rates = sweep_features["avg_rate"].values
    sorter = np.argsort(amps)
    return amps[sorter], rates[sorter]


def fi_shift(c1_amps, c1_rates, c2_amps, c2_rates):
    """Estimate the shift between two f-I curves

    Parameters
    ----------
    c1_amps: array
        Stimulus amplitudes of first set of sweeps

    c1_rates: array
        Firing rates of first set of sweeps

    c2_amps: array
        Stimulus amplitudes of second set of sweeps

    c2_rates: array
        Firing rates of second set of sweeps

    Returns
    -------
    amp_shift: float
        Estimated shift of f-I curve along amplitude axis
    """
    # Linear fit to original fI curve
    if np.all(c1_rates == 0):
        # First f-I curve is all zero, so can't figure out shift
        return np.nan
    if np.all(c2_rates == 0):
        # Second f-I curve is all zero, so can't figure out shift
        return np.nan

    # Fit f-I curve slope including largest-amplitude subthreshold point
    last_zero_index = np.flatnonzero(c1_rates)[0] - 1
    A = np.vstack([c1_amps[last_zero_index:],
                   np.ones_like(c1_amps[last_zero_index:])]).T
    m, c = np.linalg.lstsq(A, c1_rates[last_zero_index:], rcond=None)[0]

    # Relative error of later traces to best-fit line
    mask = c2_fi_rates > 0 # Can't assess zero firing rate points
    amp_shift = c2_amps[mask] - (c2_rates[mask] - c) / m
    return amp_shift.mean()


def estimate_fi_shift(core_1_lsq, core_1_start, core_1_end,
        core_2_lsq, core_2_start, core_2_end):
    """Estimate the amount the f-I curve shifted between Core 1 and Core 2 sweeps

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

    Returns
    -------
    amp_shift: float
        Estimated f-I shift between Core 2 and Core 1 sweeps
    """

    # Calculate f-I curves using same duration for both types of sweeps
    c1_dur = core_1_end - core_1_start
    c2_dur = core_2_end - core_2_start
    common_dur = min(c1_dur, c2_dur)

    c1_fi_amps, c1_fi_rates = fi_curve(core_1_lsq, core_1_start, core_1_start + common_dur)
    c2_fi_amps, c2_fi_rates = fi_curve(core_2_lsq, core_2_start, core_2_start + common_dur)

    return fi_shift(c1_fi_amps, c1_fi_rates, c2_fi_amps, c2_fi_rates)

