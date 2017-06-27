import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.model_selection as ms

def main():
    parser = argparse.ArgumentParser(description='Select final model from optimization runs')
    parser.add_argument('input_json', type=str)
    parser.add_argument('output_json', type=str)
    args = parser.parse_args()

    input = ju.read(args.input_json)

    swc_path = input["paths"]["swc"]
    fit_style_paths = input["paths"]["fit_styles"]
    best_fit_json_path = input["paths"]["best_fit_json_path"] 
    passive = ju.read(input["paths"]["passive_results"])
    preprocess = ju.read(input["paths"]["preprocess_results"])


    fits = input["paths"]["fits"]
    fit_results = ms.fit_info(fits)
    best_fit = ms.select_model(fit_results, input["paths"], passive, preprocess["v_baseline"],
                               input["noise_1_sweeps"], input["noise_2_sweeps"])
    if best_fit is None:
        raise Exception("Failed to find acceptable optimized model")

    fit_style_data = ju.read(input["paths"]["fit_styles"][best_fit["fit_type"]])
    fit_data = ms.build_fit_data(best_fit["params"], passive, preprocess, fit_style_data)
    
    ju.write(fit_json_path, fit_data)

    output = {
        "paths": {
            "fit_json": fit_json_path,
        }
    }

    ju.write(args.output_json, output)

if __name__ == "__main__": main()
    
