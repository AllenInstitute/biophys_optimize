#!/usr/bin/env python

import json
import argparse
import os.path
from pkg_resources import resource_filename


parser = argparse.ArgumentParser(description='hack in paths that strategy will do - passive')
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
parser.add_argument('passive_fit_type', type=str)
args = parser.parse_args()

fit1_handles = [
    "passive/fixnseg.hoc",
    "passive/iclamp.ses",
    "passive/params.hoc",
    "passive/mrf.ses",
]

fit2_handles = [
    "passive/fixnseg.hoc",
    "passive/iclamp.ses",
    "passive/params.hoc",
    "passive/mrf2.ses",
]

fit3_handles = [
    "passive/fixnseg.hoc",
    "passive/circuit.ses",
    "passive/params.hoc",
    "passive/mrf3.ses",
]

with open(args.input, "r") as f:
    data = json.load(f)

import biophys_optimize.passive_fitting
bopf_name = biophys_optimize.passive_fitting.__name__

new_path_info = {
    "fit1": [resource_filename(bopf_name, f) for f in fit1_handles],
    "fit2": [resource_filename(bopf_name, f) for f in fit2_handles],
    "fit3": [resource_filename(bopf_name, f) for f in fit3_handles],
    "passive_fit_results_file": os.path.join(data["paths"]["storage_directory"], "%s_results.json" % args.passive_fit_type)
}

data["paths"].update(new_path_info)
data["passive_fit_type"] = args.passive_fit_type

with open(args.output, "w") as f:
    json.dump(data, f, indent=2)
