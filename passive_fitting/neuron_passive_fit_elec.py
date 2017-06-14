#!/usr/bin/env python

import lims_utils
from neuron import h
import numpy as np
import argparse
import pkg_resources


def load_morphology(filename):
    swc = h.Import3d_SWC_read()
    swc.input(filename)
    imprt = h.Import3d_GUI(swc, 0)
    h("objref this")
    imprt.instantiate(h.this)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze cap check sweep')
    parser.add_argument('upfile', type=str)
    parser.add_argument('downfile', type=str)
    parser.add_argument('limit', type=float)
    parser.add_argument("bridge", type=float)
    parser.add_argument("elec_cap", type=float)
    parser.add_argument("swc", type=str)

    args = parser.parse_args()

    swc_path = args.swc
    up_data = np.loadtxt(args.upfile)
    down_data = np.loadtxt(args.downfile)

    h.load_file("stdgui.hoc")
    h.load_file("import3d.hoc")
    load_morphology(swc_path)

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

    load_file_list = [
        "passive/fixnseg.hoc",
        "passive/circuit.ses",
        "passive/params.hoc",
        "passive/mrf3.ses",
    ]

    for filename in load_file_list:
        file_path = pkg_resources.resource_filename(__name__, filename)
        h.load_file(file_path)

    h.v_init = 0
    h.tstop = 100
    h.cvode_active(1)

    fit_start = 4.0025

    circuit = h.LinearCircuit[0]
    circuit.R2 = args.bridge / 2.0
    circuit.R3 = args.bridge / 2.0
    circuit.C4 = args.elec_cap * 1e-3

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
    fit0.boundary.x[0] = fit_start
    fit0.boundary.x[1] = args.limit
    fit0.set_w()

    gen1 = mrf.p.pf.generatorlist.object(1)
    gen1.toggle()
    fit1 = gen1.gen.fitnesslist.object(0)

    down_t = h.Vector(down_data[:, 0])
    down_v = h.Vector(down_data[:, 1])
    fit1.set_data(down_t, down_v)
    fit1.boundary.x[0] = fit_start
    fit1.boundary.x[1] = args.limit
    fit1.set_w()

    minerr = 1e12
    for i in range(3):
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

    h.region_areas()
    print "Ri ", fit_Ri
    print "Cm ", fit_Cm
    print "Rm ", fit_Rm
    print "Final error ", minerr