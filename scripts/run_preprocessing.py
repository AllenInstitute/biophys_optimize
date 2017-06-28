import argparse
import pandas as pd
import os

import json_module as jm
import marshmallow as mm

from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.core.json_utilities as ju

from biophys_optimize.preprocess import preprocess

class PreprocessorPaths(mm.Schema):
    nwb = jm.InputFile(description="path to input NWB file")
    swc = jm.InputFile(description="path to input SWC file")
    storage_directory = jm.InputDir(description="path to storage directory")

class PreprocessorSweeps(mm.Schema):
    core_1_long_squares = mm.fields.List(mm.fields.Int, description="list of core 1 long square sweep numbers")
    core_2_long_squares = mm.fields.List(mm.fields.Int, description="list of core 2 long square sweep numbers")
    seed_1_noise = mm.fields.List(mm.fields.Int, description="list of seed 1 noise sweep numbers")
    seed_2_noise = mm.fields.List(mm.fields.Int, description="list of seed 2 noise sweep numbers")
    cap_checks = mm.fields.List(mm.fields.Int, description="list of capacitance check sweep numbers")

class PreprocessorParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(PreprocessorPaths)
    dendrite_type_tag = mm.fields.Str(description="dendrite type tag")
    sweeps = mm.fields.Nested(PreprocessorSweeps)
    bridge_avg = mm.fields.Float(description="average bridge balance")
    
def main():
    """Main sequence of pre-processing and passive fitting"""

    module = jm.JsonModule(schema_type=PreprocessorParameters)

    nwb_path = module.args["paths"]["nwb"]
    swc_path = module.args["paths"]["swc"]
    storage_directory = module.args["paths"]["storage_directory"]

    paths, results, passive_info, tasks = \
        preprocess(data_set=NwbDataSet(nwb_path),
                   swc_data=pd.read_table(swc_path, sep='\s+', comment='#', header=None),
                   dendrite_type_tag=module.args["dendrite_type_tag"],
                   sweeps=module.args["sweeps"],
                   bridge_avg=module.args["bridge_avg"],
                   storage_directory=storage_directory)

    preprocess_results_path = os.path.join(storage_directory, "preprocess_results.json")
    ju.write(preprocess_results_path, results)

    passive_info_path = os.path.join(storage_directory, "passive_info.json")
    ju.write(passive_info_path, passive_info)

    paths.update({
        "swc": swc_path,
        "nwb": nwb_path,
        "storage_directory": storage_directory,
        "preprocess_results": preprocess_results_path,
        "passive_info": passive_info_path,
    })

    output = {
        "paths": paths,
        "task_list": tasks,
    }

    ju.write(module.args["output_json"], output)

if __name__ == "__main__": main()
