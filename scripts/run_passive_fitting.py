import argparse
import allensdk.core.json_utilities as ju
import numpy as np
import biophys_optimize.neuron_passive_fit as npf

def main():
    parser = argparse.ArgumentParser(description='analyze cap check sweep')
    parser.add_argument('input_json', type=str)
    parser.add_argument('output_json', type=str)
    args = parser.parse_args()

    input = ju.read(args.input_json)

    swc_path = input["paths"]["swc"].encode('ascii', 'ignore')
    up_data = np.loadtxt(input["paths"]["up"])
    down_data = np.loadtxt(input["paths"]["down"])
    passive_fit_type = input["passive_fit_type"]
    results_file = input["paths"]["passive_fit_results_file"]

    info = ju.read(input["paths"]["passive_info"])

    npf.initialize_neuron(swc_path, input["paths"]["fit"])

    if passive_fit_type == npf.PASSIVE_FIT_1:
        results = npf.passive_fit_1(info, up_data, down_data)        
    elif passive_fit_type == npf.PASSIVE_FIT_2:
        results = npf.passive_fit_2(info, up_data, down_data)
    elif passive_fit_type == npf.PASSIVE_FIT_ELEC:
        results = npf.passive_fit_elec(info, up_data, down_data)
    else:
        raise Exception("unknown passive fit type: %s" % passive_fit_type)

    ju.write(results_file, results)

    ju.write(args.output_json, { "paths": { passive_fit_type: results_file } })


if __name__ == "__main__": main()
