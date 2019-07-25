import argparse
import pandas as pd
import os
import argschema as ags

import allensdk.core.json_utilities as ju
from ipfx.nwb_reader import create_nwb_reader
import biophys_optimize.preprocess as preprocess
from biophys_optimize.step_analysis import StepAnalysis
from biophys_optimize.sweep_functions import sweeps_from_nwb


class PreprocessorPaths(ags.schemas.DefaultSchema):
    nwb = ags.fields.InputFile(description="path to input NWB (Neurodata Without Borders) file")
    swc = ags.fields.InputFile(description="path to input SWC (morphology) file")
    storage_directory = ags.fields.OutputDir(description="path to storage directory")


class PreprocessorSweeps(ags.schemas.DefaultSchema):
    core_1_long_squares = ags.fields.List(ags.fields.Int, description="list of core 1 long square sweep numbers")
    core_2_long_squares = ags.fields.List(ags.fields.Int, description="list of core 2 long square sweep numbers")
    seed_1_noise = ags.fields.List(ags.fields.Int, description="list of seed 1 noise sweep numbers")
    seed_2_noise = ags.fields.List(ags.fields.Int, description="list of seed 2 noise sweep numbers")
    cap_checks = ags.fields.List(ags.fields.Int, description="list of capacitance check sweep numbers")


class PreprocessorParameters(ags.ArgSchema):
    paths = ags.fields.Nested(PreprocessorPaths)
    dendrite_type = ags.fields.Str(description="dendrite type (spiny or aspiny)")
    sweeps = ags.fields.Nested(PreprocessorSweeps)
    bridge_avg = ags.fields.Float(description="average bridge balance")


def main(paths, sweeps, dendrite_type, bridge_avg, output_json, **kwargs):
    """Main sequence of pre-processing and passive fitting"""

    # Extract Sweep objects (from IPFX package) from NWB file
    nwb_path = paths["nwb"] # nwb - neurodata without borders (ephys data)
    nwb_data = create_nwb_reader(nwb_path)
    core_1_lsq, c1_start, c1_end = sweeps_from_nwb(
        nwb_data, sweeps["core_1_long_squares"])
    core_2_lsq, c2_start, c2_end = sweeps_from_nwb(
        nwb_data, sweeps["core_2_long_squares"])

    # Choose sweeps to train the model
    sweep_set_to_fit, start, end = preprocess.select_core_1_or_core_2_sweeps(
        core_1_lsq, c1_start, c1_end,
        core_2_lsq, c2_start, c2_end)
    if len(sweep_set_to_fit.sweeps) == 0:
        ju.write(output_json, { 'error': "No usable sweeps found" })
        return

    # Calculate the target features from the training sweeps
    step_analysis = StepAnalysis(start, end)
    step_analysis.analyze(sweep_set_to_fit)
    target_info = preprocess.target_features(
        step_analysis.sweep_features(),
        step_analysis.spikes_data())


    # Determine maximum current used for depolarization block checks
    # during optimization

    # Loading noise sweeps to check highest current used
    noise_1, _, _ = sweeps_from_nwb(
        nwb_data, sweeps["seed_1_noise"])
    noise_2, _, _ = sweeps_from_nwb(
        nwb_data, sweeps["seed_2_noise"])

    max_i = preprocess.max_i_for_depol_block_check(
        core_1_lsq, core_2_lsq, noise_1, noise_2)

    print(max_i)
    # Prepare for passive fitting
    return

    swc_path = paths["swc"] # swc - morphology data
    storage_directory = paths["storage_directory"]

    try:
        paths, results, passive_info, s1_tasks, s2_tasks = \
            preprocess(data_set=None,
                       swc_data=pd.read_table(swc_path, sep='\s+', comment='#', header=None),
                       dendrite_type_tag=module.args["dendrite_type_tag"],
                       sweeps=module.args["sweeps"],
                       bridge_avg=module.args["bridge_avg"],
                       storage_directory=storage_directory)
    except NoUsableSweepsException as e:
        pass


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
        "stage_1_task_list": s1_tasks,
        "stage_2_task_list": s2_tasks,
    }

    ju.write(module.args["output_json"], output)

if __name__ == "__main__":
    # argschema reads arguments from a JSON file or command line arguments
    module = ags.ArgSchemaParser(schema_type=PreprocessorParameters, logger_name=None)
    main(**module.args)
