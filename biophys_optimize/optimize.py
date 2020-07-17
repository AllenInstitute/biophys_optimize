#!/usr/bin/env python

from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from mpi4py import MPI

import argparse
import numpy as np
import random
import json
import os.path
from dataclasses import dataclass
from deap import algorithms, base, creator, tools
from .utils import Utils
from .environment import NeuronEnvironment
from . import neuron_parallel


# Constants
BOUND_LOWER, BOUND_UPPER = 0.0, 1.0
DEFAULT_NGEN = 500
DEFAULT_MU = 1200

# Globals - can't be used in partial functions with NEURON
# so cannot be just passed into evaluation function
utils = None
t_vec = None
v_vec = None
i_vec = None


@dataclass
class StimParams:
    """ Dataclass for stimulation parameters

    Attributes
    ----------
    amplitude : float
        Stimulus amplitude (nA)
    delay : float
        Delay before stimulus start (ms)
    duration : float
        Duration of stimulus (ms)
    """
    amplitude: float
    delay: float
    duration: float


def eval_param_set(params, do_block_check, max_stim_amp, stim_params,
        features, targets, t_vec):
    """Calculate a fitness score (lower is better) for a set of model parameters

    Parameters
    ----------
    params : array
        Normalized (from 0 to 1) parameters for model
    do_block_check : bool
        Whether to check for depolarization block
    max_stim_amp : float
        Stimulation amplitude to use for depolarization block check
    stim_params : :class:`StimParams`
        Stimulation parameters
    features : list
        List of feature names (str)
    targets : dict
        Dictionary with feature names as keys and elements as dicts with "mean" and "stdev" key-value pairs
    t_vec : NEURON Vector
        time vector

    Returns
    -------
    float
        Sum of feature errors for use as a fitness score
    """
    utils.set_normalized_parameters(params)
    h = utils.h
    h.finitialize()
    h.run()

    try:
        feature_errors = utils.calculate_feature_errors(t_vec.as_numpy(),
                                                        v_vec.as_numpy(),
                                                        i_vec.as_numpy(),
                                                        features,
                                                        targets)
    except:
        print("Error with parameter set: %s" % str(params))
        raise

    min_fail_penalty = 75.0
    if do_block_check and np.sum(feature_errors) < min_fail_penalty * len(feature_errors):
        if check_for_block(max_stim_amp, stim_params):
            feature_errors = min_fail_penalty * np.ones_like(feature_errors)
        # Reset the stimulus back
        utils.set_iclamp_params(stim_params.amplitude, stim_params.delay,
            stim_params.duration)

    return [np.sum(feature_errors)]


def check_for_block(max_stim_amp, stim_params):
    """Check for presence of depolarization block at high stimulus amplitudes

    Parameters
    ----------
    max_stim_amp : float
        Stimulus amplitude for depolarization block check (nA)
    stim_params : :class:`StimParams`
        Stimulus parameters (only delay and duration used)


    Returns
    -------
    bool
        Whether the model exhibits depolarization block
    """

    utils.set_iclamp_params(max_stim_amp, stim_params.delay,
        stim_params.duration)
    h = utils.h
    h.finitialize()
    h.run()

    v = v_vec.as_numpy()
    t = t_vec.as_numpy()
    stim_start_idx = np.flatnonzero(t >= utils.stim.delay)[0]
    stim_end_idx = np.flatnonzero(t >= utils.stim.delay + utils.stim.dur)[0]
    depol_block_threshold = -50.0 # mV
    block_min_duration = 50.0 # ms
    long_hyperpol_threshold = -75.0 # mV

    bool_v = np.array(v > depol_block_threshold, dtype=int)
    up_indexes = np.flatnonzero(np.diff(bool_v) == 1)
    down_indexes = np.flatnonzero(np.diff(bool_v) == -1)
    if len(up_indexes) > len(down_indexes):
        down_indexes = np.append(down_indexes, [stim_end_idx])

    if len(up_indexes) == 0:
        # if it never gets high enough, that's not a good sign (meaning no spikes)
        return True
    else:
        max_depol_duration = np.max([t[down_indexes[k]] - t[up_idx] for k, up_idx in enumerate(up_indexes)])
        if max_depol_duration > block_min_duration:
            return True

    bool_v = np.array(v > long_hyperpol_threshold, dtype=int)
    up_indexes = np.flatnonzero(np.diff(bool_v) == 1)
    down_indexes = np.flatnonzero(np.diff(bool_v) == -1)
    down_indexes = down_indexes[(down_indexes > stim_start_idx) & (down_indexes < stim_end_idx)]
    if len(down_indexes) != 0:
        up_indexes = up_indexes[(up_indexes > stim_start_idx) & (up_indexes < stim_end_idx) & (up_indexes > down_indexes[0])]
        if len(up_indexes) < len(down_indexes):
            up_indexes = np.append(up_indexes, [stim_end_idx])
        max_hyperpol_duration = np.max([t[up_indexes[k]] - t[down_idx] for k, down_idx in enumerate(down_indexes)])
        if max_hyperpol_duration > block_min_duration:
            return True
    return False


def uniform(lower, upper, size=None):
    """Population generation with uniformly random variables

    Parameters
    ----------
    lower : float
        Lower bound for parameter
    upper : float
        Upper bound for parameter
    size : int, optional
        Number of samples to generate. If None, generates one sample

    Returns
    -------
    list
        Returns list with ``size`` samples uniformly distributed between ``lower`` and ``upper``
    """

    if size is None:
        return [random.uniform(a, b) for a, b in zip(lower, upper)]
    else:
        return [random.uniform(a, b) for a, b in zip([lower] * size, [upper] * size)]


def best_sum(d):
    """ Return the lowest sum across columns

    Parameters
    ----------
    d : (n, p) array
        Array with `n` samples each with `p` feature errors

    Returns
    -------
    float
        Minimum feature error sum across samples in `d`
    """
    return np.sum(d, axis=1).min()


def initPopulation(pcls, ind_init, popfile):
    """ Load starting population from file

    Parameters
    ----------
    pcls : function
        Population creation function (e.g. `list`)
    ind_init : function
        Individual initialization function
    popfile : str
        Path to population text file (with actual parameter values)

    Returns
    -------
    any
        Result of `pcls` call
    """
    popdata = np.loadtxt(popfile)
    return pcls(ind_init(utils.normalize_actual_parameters(line)) for line in popdata.tolist())


def optimize(hoc_files, compiled_mod_library, morphology_path,
             features, targets, stim_params, passive_results,
             fit_type, fit_style_data,
             ngen, seed, mu,
             storage_directory,
             starting_population=None):
    """ Perform embarrassingly parallel evolutionary optimization with NEURON

    Parameters
    ----------
    hoc_files : list
        List of hoc files for NEURON to load
    compiled_mod_library : str
        Path to compiled .mod file library
    morphology_path : str
        Path to morphology SWC file
    features : list
        List of features to match
    targets : dict
        Dictionary with feature names as keys and elements as dicts with "mean" and "stdev" key-value pairs
    stim_params : :class:`StimParams`
        Stimulation parameters
    passive_results : dict
        Dictionary with passive parameters
    fit_type : str
        Code for fit type (for output filenames)
    fit_style_data : dict
        Dictionary of fit style parameters
    ngen : int
        Number of generations
    seed : int
        Seed for random number generator
    mu : int
        Size of each generation
    storage_directory : str
        Path to storage directory
    starting_population : str, optional
        Path to file with starting population. If `None`, a random starting population
        is generated.

    Returns
    -------
    dict
        Dictionary with paths to output files
    """
    # need to be global since cannot pass via partial functions to
    # parallel NEURON mapping function
    global utils
    global v_vec, i_vec, t_vec

    do_block_check = fit_style_data["check_depol_block"]
    if do_block_check:
        max_stim_amp = preprocess_results["max_stim_test_na"]
        if max_stim_amp <= stim_params["amplitude"]:
            print("Depol block check not necessary")
            do_block_check = False
    else:
        max_stim_amp = None

    environment = NeuronEnvironment(hoc_files, compiled_mod_library)
    utils = Utils()
    h = utils.h

    utils.generate_morphology(morphology_path)
    utils.load_cell_parameters(passive_results,
                               fit_style_data["conditions"],
                               fit_style_data["channels"],
                               fit_style_data["addl_params"])
    utils.insert_iclamp()
    utils.set_iclamp_params(stim_params["amplitude"], stim_params["delay"],
                            stim_params["duration"])

    h.tstop = stim_params["delay"] * 2.0 + stim_params["duration"]
    h.celsius = fit_style_data["conditions"]["celsius"]
    h.v_init = preprocess_results["v_baseline"]
    h.cvode_active(1)
    h.cvode.atolscale("cai", 1e-4)
    h.cvode.maxstep(10)

    v_vec, i_vec, t_vec = utils.record_values()

    try:
        neuron_parallel._runworker()

        print("Setting up GA")
        random.seed(seed)

        cxpb = 0.1
        mtpb = 0.35
        eta = 10.0

        ndim = len(fit_style_data["channels"]) + len(fit_style_data["addl_params"])

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register("attr_float", uniform, BOUND_LOWER, BOUND_UPPER, ndim)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", eval_param_set,
            do_block_check=do_block_check,
            max_stim_amp=max_stim_amp,
            stim_params=stim_params,
            features=features,
            targets=targets
            )
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOWER, up=BOUND_UPPER,
            eta=eta)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOWER, up=BOUND_UPPER,
            eta=eta, indpb=mtpb)
        toolbox.register("variate", algorithms.varAnd)
        toolbox.register("select", tools.selBest)
        toolbox.register("map", neuron_parallel.map)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("best", best_sum)

        logbook = tools.Logbook()
        logbook.header = "gen", "nevals", "min", "max", "best"

        if starting_population is not None:
            print("Using a pre-defined starting population")
            toolbox.register("population_start", initPopulation, list, creator.Individual)
            pop = toolbox.population_start(starting_population)
        else:
            pop = toolbox.population(n=mu)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof = tools.HallOfFame(mu)
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        print("Best so far:")
        print(utils.actual_parameters_from_normalized(hof[0]))

        for gen in range(1, ngen + 1):
            offspring = toolbox.variate(pop, toolbox, cxpb, 1.0)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            hof.update(offspring)

            pop[:] = toolbox.select(pop + offspring, mu)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            print("Best so far:")
            print(utils.actual_parameters_from_normalized(hof[0]))

            utils.set_normalized_parameters(hof[0])
            h.finitialize()
            h.run()
            feature_errors = utils.calculate_feature_errors(t_vec.as_numpy(),
                                                            v_vec.as_numpy(),
                                                            i_vec.as_numpy(),
                                                            features,
                                                            targets)
            print("Error vector for best so far: " + str(feature_errors))

        prefix = "{:s}_{:d}_".format(fit_type, seed)

        final_pop_path = os.path.join(storage_directory, prefix + "final_pop.txt")
        np.savetxt(final_pop_path, np.array(list(map(utils.actual_parameters_from_normalized, pop)), dtype=np.float64))

        final_pop_fit_path = os.path.join(storage_directory, prefix + "final_pop_fit.txt")
        np.savetxt(final_pop_fit_path, np.array([ind.fitness.values for ind in pop]))

        final_hof_path = os.path.join(storage_directory, prefix + "final_hof.txt")
        np.savetxt(final_hof_path, np.array(list(map(utils.actual_parameters_from_normalized, hof)), dtype=np.float64))

        final_hof_fit_path = os.path.join(storage_directory, prefix + "final_hof_fit.txt")
        np.savetxt(final_hof_fit_path, np.array([ind.fitness.values for ind in hof]))

        output = {
            "paths": {
                "final_pop": final_pop_path,
                "final_pop_fit": final_pop_fit_path,
                "final_hof": final_hof_path,
                "final_hof_fit_path": final_hof_fit_path,
            }
        }

        neuron_parallel._done()

        return output
    except:
        print("Exception encountered during parallel NEURON execution")
        MPI.COMM_WORLD.Abort()
        raise



