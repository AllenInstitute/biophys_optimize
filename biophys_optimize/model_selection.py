#!/usr/bin/env python

from __future__ import print_function
from builtins import range
import argparse
import json
import os.path
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from allensdk.core.nwb_data_set import NwbDataSet
from biophys_optimize.utils import Utils
from biophys_optimize import sweep_functions as sf


def select_model(fit_results, path_info, passive, v_init, noise_1_sweeps,
                 noise_2_sweeps, max_attempts=20):
    """Choose model with best error that does not exhibit depolarization block
    on noise sweeps

    Parameters
    ----------
    fit_results : list
        List of dictionaries containing individual model fit information
    path_info : dict
        Dictionary of file paths
    passive : dict
        Dictionary of passive parameters
    v_init : float
        Initial voltage for stimulation (mV)
    noise_1_sweeps, noise_2_sweeps : list
        List of sweep numbers that used "Noise 1" and "Noise 2" protocols. Used to look
        for depolarization block that was not evoked by high-amplitude step currents.
    max_attempts : int, optional
        Number of models to evaluate before giving up

    Returns
    -------
    dict
        Dictionary with best-fit model information
    """

    errs = np.array([d["err"] for d in fit_results])
    sorted_order = np.argsort(errs)
    if len(noise_1_sweeps) == 0 and len(noise_2_sweeps) == 0:
        print("No noise stimulus available to test - selecting the model with lowest error")
        return fit_results[sorted_order[0]]

    nwb_path = path_info["nwb"]
    swc_path = path_info["swc"]

    fit_style_data = {}
    for fit_type in path_info["fit_styles"]:
        with open(path_info["fit_styles"][fit_type], "r") as f:
            fit_style_data[fit_type] = json.load(f)

    data_set = NwbDataSet(nwb_path)
    noise_stim = []
    max_t = 0
    dt = 0
    if len(noise_1_sweeps) > 0:
        v, i, t = sf.get_sweep_v_i_t_from_set(data_set, noise_1_sweeps[-1])
        i *= 1e-3 # to nA
        noise_stim.append(i)
        if np.max(t) > max_t:
            max_t = np.max(t)
        dt = t[1] - t[0]

    if len(noise_2_sweeps) > 0:
        v, i, t = sf.get_sweep_v_i_t_from_set(data_set, noise_2_sweeps[-1])
        i *= 1e-3 # to nA
        noise_stim.append(i)
        if np.max(t) > max_t:
            max_t = np.max(t)
        dt = t[1] - t[0]
    max_t *= 1e3 # to ms
    dt *= 1e3 # to ms
    print("Max t = ", max_t)

    # Set up
    if max_attempts > len(sorted_order):
        max_attempts = len(sorted_order)

    for ind in sorted_order[:max_attempts]:
        print("Testing model ", ind)

        fit = fit_results[ind]
        depol_okay = True

        utils = Utils(path_info["hoc_files"],
                      path_info["compiled_mod_library"])
        h = utils.h
        utils.generate_morphology(swc_path)
        utils.load_cell_parameters(passive,
                                   fit_style_data[fit["fit_type"]]["conditions"],
                                   fit_style_data[fit["fit_type"]]["channels"],
                                   fit_style_data[fit["fit_type"]]["addl_params"])
        utils.insert_iclamp()
        utils.set_iclamp_params(0, 0, 1e12)

        h.tstop = max_t
        h.celsius = fit_style_data[fit["fit_type"]]["conditions"]["celsius"]
        h.v_init = v_init
        h.dt = dt
        h.cvode.atolscale("cai", 1e-4)
        h.cvode.maxstep(10)
        v_vec, i_vec, t_vec = utils.record_values()

        for i in noise_stim:
            i_stim_vec = h.Vector(i)
            i_stim_vec.play(utils.stim._ref_amp, dt)
            utils.set_actual_parameters(fit["params"])
            print("Starting run")
            h.finitialize()
            h.run()
            print("Finished run")
            i_stim_vec.play_remove()
            if has_noise_block(v_vec.as_numpy(), t_vec.as_numpy()):
                depol_okay = False

        if depol_okay:
            print("Did not detect depolarization block on noise traces")
            return fit

    print("Failed to find model after looking at best {:d} organisms".format(max_attempts))
    return None


def build_fit_data(genome_vals, passive, preprocess, fit_style_info):
    """ Create dictionary for saving model info in JSON format

    Parameters
    ----------
    genome_vals : list-like
        List of numerical model parameters
    passive : dict
        Dictionary of passive parameters
    preprocess : dict
        Dictionary with pre-processing output
    fit_style_info :
        Dictionary with fit style parameters

    Returns
    -------
    dict
        Dictionary of model parameters in standard format
    """
    json_data = {}

    # passive
    json_data["passive"] = [{}]
    json_data["passive"][0]["ra"] = passive["ra"]
    json_data["passive"][0]["e_pas"] = passive["e_pas"]
    json_data["passive"][0]["cm"] = []
    for k in passive["cm"]:
        json_data["passive"][0]["cm"].append({"section": k, "cm": passive["cm"][k]})

    # fitting
    json_data["fitting"] = [{}]
    json_data["fitting"][0]["sweeps"] = preprocess["sweeps_to_fit"]
    json_data["fitting"][0]["junction_potential"] = preprocess["junction_potential"]

    # conditions
    json_data["conditions"] = [fit_style_info["conditions"]]
    json_data["conditions"][0]["v_init"] = passive["e_pas"]

    # genome
    all_params = fit_style_info["channels"] + fit_style_info["addl_params"]
    json_data["genome"] = []
    for i, p in enumerate(all_params):
        if len(p["mechanism"]) > 0:
            param_name = p["parameter"] + "_" + p["mechanism"]
        else:
            param_name = p["parameter"]
        json_data["genome"].append({"value": genome_vals[i],
                                    "section": p["section"],
                                    "name": param_name,
                                    "mechanism": p["mechanism"]
                                    })

    return json_data


def has_noise_block(v, t, depol_block_threshold=-50.0, block_min_duration = 50.0):
    """ Determine whether the model exhibits depolarization block with noise stimuli

    Parameters
    ----------
    v : array-like
        Voltage response of model
    t : array-like
        Time points for `v`
    depol_block_threshold : float, optional
        Minimum value to identify depolarized periods in `v`
    block_min_duration : float, optional
        Minimum value of depolarized period to count as depolarization block (in ms)

    Returns
    -------
    bool
        Whether depolarization block was found
    """
    stim_start_idx = 0
    stim_end_idx = len(t) - 1
    bool_v = np.array(v > depol_block_threshold, dtype=int)
    up_indexes = np.flatnonzero(np.diff(bool_v) == 1)
    down_indexes = np.flatnonzero(np.diff(bool_v) == -1)
    if len(up_indexes) > len(down_indexes):
        down_indexes = np.append(down_indexes, [stim_end_idx])

    if len(up_indexes) != 0:
        max_depol_duration = np.max([t[down_indexes[k]] - t[up_idx] for k, up_idx in enumerate(up_indexes)])
        if max_depol_duration > block_min_duration:
            print("Encountered depolarization block")
            return True

    return False


def fit_info(fits):
    """ Create list with individual model info as dictionaries

    Parameters
    ----------
    fits : dict
        Dictionary with fitting output path information

    Returns
    -------
    list
        List of model information dictionaries
    """
    info = []
    for fit in fits:
        fit_type = fit["fit_type"]
        hof_fit = np.loadtxt(fit["hof_fit"])
        hof = np.loadtxt(fit["hof"])
        for i in range(len(hof_fit)):
            info.append({
                "fit_type": fit_type,
                "err": hof_fit[i],
                "params": hof[i, :],
            })
    return info


