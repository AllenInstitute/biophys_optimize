import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.model_selection as ms

import json_module as jm
import marshmallow as mm

class ModelFitStyles(mm.Schema):
    f6 = jm.InputFile(description="")
    f9 = jm.InputFile(description="")
    f12 = jm.InputFile(description="")
    f13 = jm.InputFile(description="")

class ModelFit(mm.fields.Dict):
    fit_type = mm.fields.Str(description="")
    hof_fit = jm.InputFile(description="")
    hof = jm.InputFile(description="")
    
class ModelSelectionPaths(mm.Schema):
    swc = jm.InputFile(description="path to swc file")
    nwb = jm.InputFile(description="path to nwb file")
    fit_styles = mm.fields.Nested(ModelFitStyles, description="")
    fits = mm.fields.List(ModelFit, description="")
    best_fit_json_path = jm.OutputFile(description="where to store best fit")
    passive_results = jm.InputFile(description="passive results file")
    preprocess_results = jm.InputFile(description="preprocess results file")
    hoc_files = mm.fields.List(mm.fields.Str, description="list of hoc files")
    compiled_mod_library = jm.InputFile(description="path to compiled mod library file")
    
class ModelSelectionParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(ModelSelectionPaths)
    noise_1_sweeps = mm.fields.List(mm.fields.Int, description="list of noise 1 sweep numbers")
    noise_2_sweeps = mm.fields.List(mm.fields.Int, description="list of noise 2 sweep numbers")

class ModelSelectionModule(jm.JsonModule):
    def __init__(self, *args, **kwargs):
        super(ModelSelectionModule, self).__init__(schema_type=ModelSelectionParameters,
                                                   *args, **kwargs)

def main():
    module = ModelSelectionModule()

    swc_path = module.args["paths"]["swc"]
    fit_style_paths = module.args["paths"]["fit_styles"]
    best_fit_json_path = module.args["paths"]["best_fit_json_path"] 
    passive = ju.read(module.args["paths"]["passive_results"])
    preprocess = ju.read(module.args["paths"]["preprocess_results"])


    fits = module.args["paths"]["fits"]
    fit_results = ms.fit_info(fits)
    best_fit = ms.select_model(fit_results, module.args["paths"], passive, preprocess["v_baseline"],
                               module.args["noise_1_sweeps"], module.args["noise_2_sweeps"])
    if best_fit is None:
        raise Exception("Failed to find acceptable optimized model")

    fit_style_data = ju.read(module.args["paths"]["fit_styles"][best_fit["fit_type"]])
    fit_data = ms.build_fit_data(best_fit["params"], passive, preprocess, fit_style_data)
    
    ju.write(best_fit_json_path, fit_data)

    output = {
        "paths": {
            "fit_json": best_fit_json_path,
        }
    }

    ju.write(module.args["output_json"], output)

if __name__ == "__main__": main()
    
