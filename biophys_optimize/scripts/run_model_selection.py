import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.model_selection as ms

import argschema as ags

class ModelFitStyles(ags.schemas.DefaultSchema):
    f6 = ags.fields.InputFile(description="")
    f9 = ags.fields.InputFile(description="")
    f12 = ags.fields.InputFile(description="")
    f13 = ags.fields.InputFile(description="")

class ModelFit(ags.schemas.DefaultSchema): 
    fit_type = ags.fields.Str(description="")
    hof_fit = ags.fields.InputFile(description="")
    hof = ags.InputFile(description="")
    
class ModelSelectionPaths(ags.schemas.DefaultSchema):
    swc = ags.fields.InputFile(description="path to swc file")
    nwb = ags.fields.InputFile(description="path to nwb file")
    fit_styles = ags.fields.Nested(ModelFitStyles, description="")

    fits = ags.fields.Nested(ModelFit, description="", many=True)

    best_fit_json_path = ags.fields.OutputFile(description="where to store best fit")
    passive_results = ags.fields.InputFile(description="passive results file")
    preprocess_results = ags.fields.InputFile(description="preprocess results file")
    hoc_files = ags.fields.List(ags.fields.Str, description="list of hoc files")
    compiled_mod_library = ags.fields.InputFile(description="path to compiled mod library file")
    
class ModelSelectionParameters(ags.ArgSchema):
    paths = ags.fields.Nested(ModelSelectionPaths)
    noise_1_sweeps = ags.fields.List(ags.fields.Int, description="list of noise 1 sweep numbers")
    noise_2_sweeps = ags.fields.List(ags.fields.Int, description="list of noise 2 sweep numbers")


def main():
    module = ags.ArgSchemaParser(schema_type=ModelSelectionParameters)

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
    
