"""
Script to select the starting population

.. autoclass:: PopulationSelectionParameters
.. autoclass:: PopulationSelectionPaths
.. autoclass:: ModelFit
"""

from __future__ import print_function
import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.population_selection as ps

import argschema as ags

class ModelFit(ags.schemas.DefaultSchema):
    fit_type = ags.fields.Str(description="")
    hof_fit = ags.fields.InputFile(description="")
    hof = ags.fields.InputFile(description="")


class PopulationSelectionPaths(ags.schemas.DefaultSchema):
    fits = ags.fields.Nested(ModelFit, description="", many=True)


class PopulationSelectionParameters(ags.ArgSchema):
    paths = ags.fields.Nested(PopulationSelectionPaths)


def main():
    module = ags.ArgSchemaParser(schema_type=PopulationSelectionParameters)
    print(module.args)

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
