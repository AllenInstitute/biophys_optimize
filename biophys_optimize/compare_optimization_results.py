#!/usr/bin/env python

import argparse
import numpy as np
import json


def main(input_file, output_file):
    with open(input_file, "r") as f:
        input = json.load(f)

    fit_types = input["fit_types"]

    best_pop = {}
    best_file = ""
    for fit_type in fit_types:
        best_pop[fit_type] = {}
        best_error = 1e12
        for seed_results in input["paths"][fit_type]:
            hof_fit_file = seed_results["hof_fit"]
            hof_fit = np.loadtxt(hof_fit_file)
            best_for_seed = np.min(hof_fit)
            if best_for_seed < best_error:
                best_file = seed_results["hof"]
                best_error = best_for_seed
        best_pop[fit_type]["best"] = best_file

    output = {
        "paths": best_pop
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a DEAP optimization run.')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    main(args.input, args.output)
