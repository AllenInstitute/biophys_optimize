import argparse
from biophys_optimize.optimize import optimize
import allensdk.core.json_utilities as ju

import json_module as jm
import marshmallow as mm

class OptimizePaths(mm.Schema):
    preprocess_results = jm.InputFile(description="path to preprocess results file")
    passive_results = jm.InputFile(description="path to passive results file")
    fit_style = jm.InputFile(description="path to fit style file")
    hoc_files = mm.fields.List(mm.fields.Str, description="list of hoc files")
    compiled_mod_library = jm.InputFile(description="path to compiled mod library file")
    swc = jm.InputFile(description="path to SWC file")
    storage_directory = mm.fields.Str(description="where to store outputs")
    starting_population = mm.fields.Str(description="starting population")

class OptimizeParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(OptimizePaths)
    fit_type = mm.fields.Str(description="fit type")
    seed = mm.fields.Int(description="seed")
    mu = mm.fields.Int(description="mu")
    ngen = mm.fields.Int(description="ngen")

def main():
    module = jm.JsonModule(schema_type=OptimizeParameters)

    preprocess_results = ju.read(module.args["paths"]["preprocess_results"])
    passive_results = ju.read(module.args["paths"]["passive_results"])
    fit_style_data = ju.read(module.args["paths"]["fit_style"])

    results = optimize(hoc_files=module.args["paths"]["hoc_files"], 
                       compiled_mod_library=module.args["paths"]["compiled_mod_library"], 
                       morphology_path=module.args["paths"]["swc"], 
                       preprocess_results=preprocess_results, 
                       passive_results=passive_results, 
                       fit_type=module.args["fit_type"], 
                       fit_style_data=fit_style_data,
                       seed=module.args["seed"], 
                       ngen=module.args["ngen"], 
                       mu=module.args["mu"],    
                       storage_directory = module.args["paths"]["storage_directory"],
                       starting_population = module.args["paths"].get("starting_population",None))

    print "writing output"
    ju.write(module.args["output_json"], results)

if __name__ == "__main__": main()
