"""
Script to run the optimization using an evolutionary algorithm.

.. autoclass:: OptimizeParameters
.. autoclass:: OptimizePaths

"""

import argparse
from biophys_optimize.optimize import optimize, StimParams
import allensdk.core.json_utilities as ju
import logging

import argschema as ags

class OptimizePaths(ags.schemas.DefaultSchema):
    preprocess_results = ags.fields.InputFile(description="path to preprocess results file")
    passive_results = ags.fields.InputFile(description="path to passive results file")
    fit_style = ags.fields.InputFile(description="path to fit style file")
    hoc_files = ags.fields.List(ags.fields.Str, description="list of hoc files")
    compiled_mod_library = ags.fields.InputFile(description="path to compiled mod library file")
    swc = ags.fields.InputFile(description="path to SWC file")
    storage_directory = ags.fields.Str(description="where to store outputs")
    starting_population = ags.fields.Str(description="starting population")

class OptimizeParameters(ags.ArgSchema):
    paths = ags.fields.Nested(OptimizePaths)
    fit_type = ags.fields.Str(description="fit type")
    seed = ags.fields.Int(description="seed")
    mu = ags.fields.Int(description="mu")
    ngen = ags.fields.Int(description="ngen")

def main():
    module = ags.ArgSchemaParser(schema_type=OptimizeParameters)

    preprocess_results = ju.read(module.args["paths"]["preprocess_results"])
    passive_results = ju.read(module.args["paths"]["passive_results"])
    fit_style_data = ju.read(module.args["paths"]["fit_style"])

    results = optimize(hoc_files=module.args["paths"]["hoc_files"],
                       compiled_mod_library=module.args["paths"]["compiled_mod_library"],
                       morphology_path=module.args["paths"]["swc"],
                       features=preprocess_results["features"],
                       targets=preprocess_results["target_features"],
                       stim_params=StimParams(preprocess_results["stimulus"]),
                       passive_results=passive_results,
                       fit_type=module.args["fit_type"],
                       fit_style_data=fit_style_data,
                       seed=module.args["seed"],
                       ngen=module.args["ngen"],
                       mu=module.args["mu"],
                       storage_directory = module.args["paths"]["storage_directory"],
                       starting_population = module.args["paths"].get("starting_population",None))

    logging.info("Writing optimization output")
    ju.write(module.args["output_json"], results)

if __name__ == "__main__": main()
