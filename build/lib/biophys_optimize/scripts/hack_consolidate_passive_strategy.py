#!/usr/bin/env python

import argparse
import os.path
import allensdk.core.json_utilities as ju
import biophys_optimize.neuron_passive_fit as npf

parser = argparse.ArgumentParser(description='hack in paths that strategy will do - passive')
parser.add_argument('preprocess_out', type=str)
parser.add_argument('fit_1_out', type=str)
parser.add_argument('fit_2_out', type=str)
parser.add_argument('fit_elec_out', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

data = ju.read(args.preprocess_out)
fit_1 = ju.read(args.fit_1_out)
fit_2 = ju.read(args.fit_2_out)
fit_3 = ju.read(args.fit_elec_out)

out_data = {
    "paths": {
        "passive_info": data["paths"]["passive_info"],
        "preprocess_results": data["paths"]["preprocess_results"],
        "passive_fit_1": fit_1["paths"][npf.PASSIVE_FIT_1],
        "passive_fit_2": fit_2["paths"][npf.PASSIVE_FIT_2],
        "passive_fit_elec": fit_3["paths"][npf.PASSIVE_FIT_ELEC],
        "passive_results": os.path.join(data["paths"]["storage_directory"], "passive_results.json")
        }
}

ju.write(args.output, out_data)



