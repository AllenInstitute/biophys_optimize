import argparse
import pandas as pd
import os

from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.core.json_utilities as ju

from biophys_optimize.preprocess import preprocess

def main():
    parser = argparse.ArgumentParser(description='Pre-process and prepare for passive fits')
    parser.add_argument('input_json', type=str)
    parser.add_argument('output_json', type=str)

    args = parser.parse_args()

    """Main sequence of pre-processing and passive fitting"""
    input = ju.read(args.input_json)

    nwb_path = input["paths"]["nwb"]
    swc_path = input["paths"]["swc"]
    storage_directory=input["paths"]["storage_directory"]

    paths, results, passive_info, tasks = \
        preprocess(data_set=NwbDataSet(nwb_path),
                   swc_data=pd.read_table(swc_path, sep='\s+', comment='#', header=None),
                   dendrite_type_tag=input["dendrite_type_tag"],
                   sweeps=input["sweeps"],
                   bridge_avg=input["bridge_avg"],
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

    ju.write(args.output_json, output)

if __name__ == "__main__": main()
