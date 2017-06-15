#!/usr/bin/env python

import json
import argparse
import logging
import os.path


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(input_file, output_file):
    with open(input_file, "r") as f:
        input = json.load(f)

    with open(input["paths"]["preprocess_results"], "r") as f:
        preprocess_results = json.load(f)
    is_spiny = preprocess_results["is_spiny"]

    # First check if fits were run
    with open(input["paths"]["passive_info"], "r") as f:
        info = json.load(f)
    if info["should_run"]:
        ra, cm1, cm2 = compare_runs(input)
    else:
        ra = 100.
        cm1 = 1.
        if is_spiny:
            cm2 = 2.
        else:
            cm2 = 1.

    passive = {
        "ra": ra,
        "cm": {"soma": cm1, "axon": cm1, "dend": cm2 },
        "e_pas": preprocess_results["v_baseline"]
    }

    passive["e_pas"] = preprocess_results["v_baseline"]
    if preprocess_results["has_apical"]:
        passive["cm"]["apic"] = cm2

    storage_directory = input["paths"]["storage_directory"]
    passive_results_path = os.path.join(storage_directory, "passive_results.json")
    with open(passive_results_path, "w") as f:
        json.dump(passive, f, indent=2)

    output = {
        "paths": {
            "passive_results": passive_results_path,
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def compare_runs(input):
    # Fit type 1 - Allows Ra, Cm, Rm to vary
    with open(input["paths"]["passive_fit1"], "r") as f:
        fit_1 = json.load(f)

    # Fit type 2 - Allows Cm, Rm to vary & fixes Ra = 100
    with open(input["paths"]["passive_fit2"], "r") as f:
        fit_2 = json.load(f)

    # Fit type 3 - Allows Ra, Cm, Rm to vary & models the recording electrode
    with open(input["paths"]["passive_fit3"], "r") as f:
        fit_3 = json.load(f)

    # Also get pre-process results
    with open(input["paths"]["preprocess_results"], "r") as f:
        preprocess_results = json.load(f)

    # Check for various fitting outcomes and pick best results
    cm_rel_delta = (fit_1["cm"] - fit_3["cm"]) / fit_1["cm"]
    if fit_2["err"] < fit_1["err"]:
        logger.info("Fixed Ri gave better results than original")
        if fit_2["err"] < fit_3["err"]:
            logger.info("Using fixed Ri results")
            fit_for_next_step = fit_2
        else:
            logger.info("Using electrode results")
            fit_for_next_step = fit_3
    elif abs(cm_rel_delta) > 0.1:
        logger.info("Original and electrode fits not in agreement")
        logger.debug("original Cm: %g", fit_1["cm"])
        logger.debug("w/ electrode Cm: %g", fit_3["cm"])
        if fit_1["err"] < fit_3["err"]:
            logger.info("Original has lower error")
            fit_for_next_step = fit_1
        else:
            logger.info("Electrode has lower error")
            fit_for_next_step = fit_3
    else:
        fit_for_next_step = fit_1

    ra = fit_for_next_step["ra"]
    is_spiny = preprocess_results["is_spiny"]
    if is_spiny:
        combo_cm = fit_for_next_step["cm"]
        a1 = fit_for_next_step["a1"]
        a2 = fit_for_next_step["a2"]
        cm1 = 1.0
        cm2 = (combo_cm * (a1 + a2) - a1) / a2
    else:
        cm1 = fit_for_next_step["cm"]
        cm2 = fit_for_next_step["cm"]

    return ra, cm1, cm2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare passive fits and determine which to use')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    main(args.input, args.output)
