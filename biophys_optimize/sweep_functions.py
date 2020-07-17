import numpy as np
from ipfx.sweep import Sweep, SweepSet
import ipfx.stim_features as stf


def sweeps_from_nwb(nwb_data, sweep_number_list):
    """ Generate a SweepSet object from an IPFX EphysDataSet and list of sweep numbers

    Sweeps should be in current-clamp mode.

    Parameters
    ----------
    nwb_data: EphysDataSet
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

    sweep_set = nwb_data.sweep_set(sweep_number_list)

    start = None
    dur = None
    if len(sweep_set.sweeps) > 0:
        first_sweep = sweep_set.sweeps[0]
        start, dur, _, _, _ = stf.get_stim_characteristics(first_sweep.i, first_sweep.t)

    if start is None or dur is None:
        return sweep_set, None, None
    else:
        return sweep_set, start, start + dur


def sweep_set_for_model(t, v, i):
    """Generate a SweepSet object based on a single model sweep

    Parameters
    ----------
    t: array
        Time data (sec)
    v: array
        Voltage data (mV)
    i: array
        Current stimulus data (nA)

    Returns
    -------
    SweepSet
        Contains one Sweep object
    """
    sampling_rate = 1 / (t[1] - t[0])
    sweep = Sweep(t=t,
                  v=v,
                  i=i,
                  sampling_rate=sampling_rate,
                  sweep_number=None,
                  clamp_mode="CurrentClamp",
                  epochs=None,
                  )
    return SweepSet([sweep])


