import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.population_selection as ps

import json_module as jm
import marshmallow as mm


class ModelFit(mm.Schema):
    fit_type = mm.fields.Str(description="")
    hof_fit = jm.InputFile(description="")
    hof = jm.InputFile(description="")


class PopulationSelectionPaths(mm.Schema):
    fits = mm.fields.Nested(ModelFit, description="", many=True)


class PopulationSelectionParameters(jm.ModuleParameters):
    paths = mm.fields.Nested(PopulationSelectionPaths)


class PopulationSelectionModule(jm.JsonModule):
    def __init__(self, *args, **kwargs):
        super(PopulationSelectionModule, self).__init__(schema_type=PopulationSelectionParameters,
                                                        *args, **kwargs)


def main():
    module = PopulationSelectionModule()

    fits = module.args["paths"]["fits"]
    populations = ps.population_info(fits)
    starting_populations = ps.select_starting_population(populations)

    output = {
        "paths": {
            "starting_populations": starting_populations,
        }
    }

    ju.write(module.args["output_json"], output)


if __name__ == "__main__": main()
