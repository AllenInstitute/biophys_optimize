#!/usr/bin/env python

import numpy as np
from ipfx.sweep import Sweep, SweepSet
import ipfx.stim_features as stf


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

