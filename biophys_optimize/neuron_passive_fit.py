#!/usr/bin/env python

from builtins import range
from neuron import h
import numpy as np
import argparse
import json
import os.path
from dataclasses import dataclass

PASSIVE_FIT_1 = "passive_fit_1"
PASSIVE_FIT_2 = "passive_fit_2"
PASSIVE_FIT_ELEC = "passive_fit_elec"

@dataclass
class PassiveFitResult:
    """DataClass for passive fit results.

    Attributes
    ----------
    ra : float
        Axial resistivity
    cm : float
        Specific membrane capacitance
    rm : float
        Specific membrane resistance
    a1 : float
        Surface area of the soma and axon compartments
    a2 : float
        Surface area of the dendrite compartments
    """
    ra: float
    cm: float
    rm: float
    err: float
    a1: float
    a2: float



def initialize_neuron(swc_path, file_paths):
    """ Initialize neuron with morphology and load hoc files

    Parameters
    ----------
    swc_path : str
        Path to SWC file
    file_paths : list
        List of str file paths of files for NEURON to load

    """
    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")

    swc = h.Import3d_SWC_read()
    swc.input(swc_path)
    imprt = h.Import3d_GUI(swc, 0)
    h("objref this")
    imprt.instantiate(h.this)

    for sec in h.allsec():
        if sec.name().startswith("axon"):
            h.delete_section(sec=sec)

    axon = h.Section()
    axon.L = 60
    axon.diam = 1
    axon.connect(h.soma[0], 0.5, 0)

    h.define_shape()

    for sec in h.allsec():
        sec.insert('pas')
        for seg in sec:
            seg.pas.e = 0

    for file_path in file_paths:
        h.load_file(file_path.encode("ascii", "ignore"))


def _common_config(fit_window_start, fit_window_end, up_data, down_data):
    h.v_init = 0
    h.tstop = 100
    h.cvode_active(1)

    v_rec = h.Vector()
    t_rec = h.Vector()
    v_rec.record(h.soma[0](0.5)._ref_v)
    t_rec.record(h._ref_t)

    mrf = h.MulRunFitter[0]
    gen0 = mrf.p.pf.generatorlist.object(0)
    gen0.toggle()
    fit0 = gen0.gen.fitnesslist.object(0)

    up_t = h.Vector(up_data[:, 0])
    up_v = h.Vector(up_data[:, 1])
    fit0.set_data(up_t, up_v)
    fit0.boundary.x[0] = fit_window_start
    fit0.boundary.x[1] = fit_window_end
    fit0.set_w()

    gen1 = mrf.p.pf.generatorlist.object(1)
    gen1.toggle()
    fit1 = gen1.gen.fitnesslist.object(0)

    down_t = h.Vector(down_data[:, 0])
    down_v = h.Vector(down_data[:, 1])
    fit1.set_data(down_t, down_v)
    fit1.boundary.x[0] = fit_window_start
    fit1.boundary.x[1] = fit_window_end
    fit1.set_w()

    return mrf


def passive_fit_1(up_data, down_data, fit_window_start, fit_window_end,
        n_init=10):
    """ Fit passive properties (Ri, Cm, and Rm)

    Parameters
    ----------
    up_data: (n_samples, 2) array
        Positive-going times and voltage responses
    down_data: (n_samples, 2) array
        Negative-going times and voltage responses
    fit_window_start: float
        Start of fit window (ms)
    fit_window_end: float
        End of fit window (ms)
    n_init: int, optional
        Number of random starts for passive fitting (default 10)

    Returns
    -------
    :class:`PassiveFitResult`
    """
    mrf = _common_config(fit_window_start, fit_window_end, up_data, down_data)

    minerr = 1e12
    for i in range(n_init):
        # Need to re-initialize the internal MRF variables, not top-level proxies
        # for randomize() to work
        mrf.p.pf.parmlist.object(0).val = 100
        mrf.p.pf.parmlist.object(1).val = 1
        mrf.p.pf.parmlist.object(2).val = 10000
        mrf.p.pf.putall()
        mrf.randomize()
        mrf.prun()
        if mrf.opt.minerr < minerr:
            fit_Ri = h.Ri
            fit_Cm = h.Cm
            fit_Rm = h.Rm
            minerr = mrf.opt.minerr

    return PassiveFitResult(
        fit_Ri, fit_Cm, fit_Rm, minerr, h.somaaxon_area(), h.alldend_area())


def passive_fit_2(up_data, down_data, fit_window_start, fit_window_end,
        n_init=10):
    """ Fit passive properties (Cm, and Rm) with Ri fixed at 100 Ohm-cm

    Parameters
    ----------
    up_data: array, shape (n_samples, 2)
        Positive-going times and voltage responses
    down_data: array, shape (n_samples, 2)
        Negative-going times and voltage responses
    fit_window_start: float
        Start of fit window (ms)
    fit_window_end: float
        End of fit window (ms)
    n_init: int (optional, default 10)
        Number of random starts for passive fitting

    Returns
    -------
    :class:`PassiveFitResult`
    """
    mrf = _common_config(fit_window_start, fit_window_end, up_data, down_data)

    minerr = 1e12
    for i in range(n_init):
        # Need to re-initialize the internal MRF variables, not top-level proxies
        # for randomize() to work
        mrf.p.pf.parmlist.object(0).val = 1
        mrf.p.pf.parmlist.object(1).val = 10000
        mrf.p.pf.putall()
        mrf.randomize()
        mrf.prun()
        if mrf.opt.minerr < minerr:
            fit_Ri = h.Ri
            fit_Cm = h.Cm
            fit_Rm = h.Rm
            minerr = mrf.opt.minerr

    return PassiveFitResult(
        fit_Ri, fit_Cm, fit_Rm, minerr, h.somaaxon_area(), h.alldend_area())


def passive_fit_elec(up_data, down_data, fit_window_start, fit_window_end,
        bridge, electrode_cap, n_init=10):
    """ Fit passive properties (Ri, Cm, and Rm) with simulated recording electrode

    Parameters
    ----------
    up_data: array, shape (n_samples, 2)
        Positive-going times and voltage responses
    down_data: array, shape (n_samples, 2)
        Negative-going times and voltage responses
    fit_window_start: float
        Start of fit window (ms)
    fit_window_end: float
        End of fit window (ms)
    bridge: float
        Value of bridge balance (estimated series resistance), MOhm
    electrode_cap: float
        Value of electrode capacitance (pF)
    n_init: int (optional, default 10)
        Number of random starts for passive fitting

    Returns
    -------
    :class:`PassiveFitResult`
    """
    mrf = _common_config(fit_window_start, fit_window_end, up_data, down_data)

    circuit = h.LinearCircuit[0]
    circuit.R2 = bridge / 2.0
    circuit.R3 = bridge / 2.0
    circuit.C4 = electrode_cap * 1e-3

    minerr = 1e12
    for i in range(n_init):
        # Need to re-initialize the internal MRF variables, not top-level proxies
        # for randomize() to work
        mrf.p.pf.parmlist.object(0).val = 100
        mrf.p.pf.parmlist.object(1).val = 1
        mrf.p.pf.parmlist.object(2).val = 10000
        mrf.p.pf.putall()
        mrf.randomize()
        mrf.prun()
        if mrf.opt.minerr < minerr:
            fit_Ri = h.Ri
            fit_Cm = h.Cm
            fit_Rm = h.Rm
            minerr = mrf.opt.minerr

    return PassiveFitResult(
        fit_Ri, fit_Cm, fit_Rm, minerr, h.somaaxon_area(), h.alldend_area())



