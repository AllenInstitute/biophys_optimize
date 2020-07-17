"""
Script to preprocess sweeps and morphologies before optimization.

.. autoclass:: PreprocessorParameters
.. autoclass:: PreprocessorSweeps
.. autoclass:: PreprocessorPaths


"""
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
    core_1_long_squares = ags.fields.List(ags.fields.Int,
        description="list of core 1 long square sweep numbers",
        cli_as_single_argument=True)
    core_2_long_squares = ags.fields.List(ags.fields.Int,
        description="list of core 2 long square sweep numbers",
        cli_as_single_argument=True)
    seed_1_noise = ags.fields.List(ags.fields.Int,
        description="list of seed 1 noise sweep numbers",
        cli_as_single_argument=True)
    seed_2_noise = ags.fields.List(ags.fields.Int,
        description="list of seed 2 noise sweep numbers",
        cli_as_single_argument=True)
    cap_checks = ags.fields.List(ags.fields.Int,
        description="list of capacitance check sweep numbers",
        cli_as_single_argument=True)


class PreprocessorParameters(ags.ArgSchema):
    paths = ags.fields.Nested(PreprocessorPaths)
    dendrite_type = ags.fields.Str(description="dendrite type (spiny or aspiny)",
        validation=lambda x: x in ["spiny", "aspiny"])
    sweeps = ags.fields.Nested(PreprocessorSweeps)
    bridge_avg = ags.fields.Float(description="average bridge balance")
    passive_fit_start_time = ags.fields.Float(description="start time of passive fit window (ms)", default=4.0)
    electrode_capacitance = ags.fields.Float(description="Capacitance of electrode for passive fitting (pF)", default=1.0)
    junction_potential = ags.fields.Float(description="Liquid junction potential (mV)", default=-14.0)
    random_seeds = ags.fields.List(ags.fields.Integer, description="list of random seeds",
        default=[1234, 1001, 4321, 1024, 2048],
        cli_as_single_argument=True)


def main(paths, sweeps, dendrite_type, bridge_avg, passive_fit_start_time,
        electrode_capacitance, junction_potential, random_seeds,
        output_json, **kwargs):
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

    stim_amp = step_analysis.sweep_features()["stim_amp"].values[0]
    stim_dur = end - start
    v_baseline = target_info.at["v_baseline", "mean"]

    # Determine maximum current used for depolarization block checks
    # during optimization

    # Load noise sweeps to check highest current used
    noise_1, _, _ = sweeps_from_nwb(
        nwb_data, sweeps["seed_1_noise"])
    noise_2, _, _ = sweeps_from_nwb(
        nwb_data, sweeps["seed_2_noise"])

    max_i = preprocess.max_i_for_depol_block_check(
        core_1_lsq, core_2_lsq, noise_1, noise_2)

    # Prepare inputs for passive fitting
    is_spiny = dendrite_type == "spiny"

    cap_checks, _, _ = sweeps_from_nwb(
        nwb_data, sweeps["cap_checks"])
    if len(cap_checks.sweeps) == 0:
        logging.info("No cap check traces found")
        should_run_passive_fit = False
        passive_info = {
            "should_run": False,
        }
    else:
        grand_up, grand_down, t = preprocess.cap_check_grand_averages(cap_checks)
        up_file, down_file = preprocess.save_grand_averages(
            grand_up, grand_down, t, paths["storage_directory"])
        escape_time = preprocess.passive_fit_window(grand_up, grand_down, t,
            start_time=passive_fit_start_time)
        passive_info = {
            "should_run": True,
            "bridge": bridge_avg,
            "fit_window_start": passive_fit_start_time,
            "fit_window_end": escape_time,
            "electrode_cap": electrode_capacitance,
            "is_spiny": is_spiny,
        }
        paths["up"] = up_file
        paths["down"] = down_file

    passive_info_path = os.path.join(
        paths["storage_directory"], "passive_info.json")
    ju.write(passive_info_path, passive_info)

    # Determine whether morphology has an apical dendrite
    has_apical = preprocess.swc_has_apical_compartments(paths["swc"])

    # Decide which fits to run based on morphology and AP width
    fit_types = preprocess.FitStyle.get_fit_types(
        has_apical=has_apical,
        is_spiny=is_spiny,
        width=target_info.at["width", "mean"])

    stage_1_tasks = [{"fit_type": fit_type, "seed": seed}
        for seed in random_seeds
        for fit_type in fit_types]

    stage_2_tasks = [{"fit_type": preprocess.FitStyle.map_stage_2(fit_type), "seed": seed}
        for seed in random_seeds
        for fit_type in fit_types]

    preprocess_results_path = os.path.join(
        paths["storage_directory"], "preprocess_results.json")
    ju.write(preprocess_results_path, {
        "is_spiny": is_spiny,
        "has_apical": has_apical,
        "junction_potential": junction_potential,
        "max_stim_test_na": max_i,
        "v_baseline": v_baseline,
        "stimulus": {
            "amplitude": 1e-3 * stim_amp, # to nA
            "delay": 1e3,
            "duration": 1e3 * stim_dur, # to ms
        },
        "target_features": target_info.to_dict(orient="index"),
        "sweeps": sweeps,
        "sweeps_to_fit": [s.sweep_number for s in sweep_set_to_fit.sweeps],
    })

    paths.update({
        "preprocess_results": preprocess_results_path,
        "passive_info": passive_info_path,
    })

    output = {
        "paths": paths,
        "stage_1_task_list": stage_1_tasks,
        "stage_2_task_list": stage_2_tasks,
    }

    ju.write(module.args["output_json"], output)


if __name__ == "__main__":
    # argschema reads arguments from a JSON file or command line arguments
    module = ags.ArgSchemaParser(schema_type=PreprocessorParameters, logger_name=None)
    main(**module.args)
