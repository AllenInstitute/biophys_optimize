import argparse

import allensdk.core.json_utilities as ju

import biophys_optimize.consolidate_passive_fits as cpf

import json_module as jm
import marshmallow as mm

class ConsolidatePaths(jm.ModuleParameters):
    preprocess_results = jm.InputFile(description="path to preprocess results file")
    passive_info = jm.InputFile(description="path to passive info file")
    passive_fit_1 = jm.InputFile(desscription="path to passive fit 1 results file")
    passive_fit_2 = jm.InputFile(desscription="path to passive fit 2 results file")
    passive_fit_elec = jm.InputFile(desscription="path to passive fit elec results file")
    passive_results = jm.OutputFile(desscription="path to store consolidated results")

class ConsolidateParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(ConsolidatePaths)

class ConsolidateModule(jm.JsonModule):
    def __init__(self, *args, **kwargs):
        super(ConsolidateModule, self).__init__(schema_type=ConsolidateParameters,
                                                *args, **kwargs)    

def main():
    module = ConsolidateModule()

    preprocess_results = ju.read(module.args["paths"]["preprocess_results"])
    is_spiny = preprocess_results["is_spiny"]
    info = ju.read(module.args["paths"]["passive_info"])

    fit_1_path = module.args["paths"]["passive_fit_1"]
    fit_1 = ju.read(fit_1_path)

    fit_2_path = module.args["paths"]["passive_fit_2"]
    fit_2 = ju.read(fit_2_path)

    fit_3_path = module.args["paths"]["passive_fit_elec"]
    fit_3 = ju.read(fit_3_path)

    if info["should_run"]:
        ra, cm1, cm2 = cpf.compare_runs(preprocess_results, fit_1, fit_2, fit_3)
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

    passive_results_path = module.args["paths"]["passive_results"]
    ju.write(passive_results_path, passive)

    output = {
        "paths": {
            "passive_results": passive_results_path,
        }
    }

    ju.write(module.args["output_json"], output)

if __name__ == "__main__": main()
    
