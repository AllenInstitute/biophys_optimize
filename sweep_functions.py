#!/usr/bin/env python

import lims_utils
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor, _step_stim_amp


C1LS_START = 1.02 # seconds
C1LS_END = 2.02 # seconds
C2LS_START = 1.02 # seconds
C2LS_END = 3.02 # seconds


def sweep_set_extractor_from_list(sweep_list, data_set, start, end, jxn=None)
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
