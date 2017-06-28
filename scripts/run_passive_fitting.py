import argparse
import allensdk.core.json_utilities as ju
import numpy as np
import biophys_optimize.neuron_passive_fit as npf

import json_module as jm
import marshmallow as mm

class PassiveFittingPaths(jm.ModuleParameters):
    swc = jm.InputFile(description="path to SWC file")
    up = jm.InputFile(descritpion="up data path")
    down = jm.InputFile(descritpion="down data path")
    passive_fit_results_file = jm.OutputFile(description="passive fit results file")
    passive_info = mm.fields.Str(description="passive info file")
    fit = mm.fields.List(jm.InputFile, description="list of passive fitting files")

class PassiveFittingParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(PassiveFittingPaths)
    passive_fit_type = mm.fields.Str(description="passive fit type")

class PassiveFittingModule(jm.JsonModule):
    def __init__(self, *args, **kwargs):
        super(PassiveFittingModule, self).__init__(schema_type=PassiveFittingParameters,
                                                    *args, **kwargs)

def main():
    module = PassiveFittingModule()
    
    swc_path = module.args["paths"]["swc"].encode('ascii', 'ignore')
    up_data = np.loadtxt(module.args["paths"]["up"])
    down_data = np.loadtxt(module.args["paths"]["down"])
    passive_fit_type = module.args["passive_fit_type"]
    results_file = module.args["paths"]["passive_fit_results_file"]

    info = ju.read(module.args["paths"]["passive_info"])

    npf.initialize_neuron(swc_path, module.args["paths"]["fit"])

    if passive_fit_type == npf.PASSIVE_FIT_1:
        results = npf.passive_fit_1(info, up_data, down_data)        
    elif passive_fit_type == npf.PASSIVE_FIT_2:
        results = npf.passive_fit_2(info, up_data, down_data)
    elif passive_fit_type == npf.PASSIVE_FIT_ELEC:
        results = npf.passive_fit_elec(info, up_data, down_data)
    else:
        raise Exception("unknown passive fit type: %s" % passive_fit_type)

    ju.write(results_file, results)

    ju.write(module.args["output_json"], { "paths": { passive_fit_type: results_file } })


if __name__ == "__main__": main()
