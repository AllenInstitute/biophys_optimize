import argparse
import allensdk.core.json_utilities as ju

import biophys_optimize.population_selection as ps

import argschema
import marshmallow as mm


class ModelFit(mm.Schema):
    fit_type = mm.fields.Str(description="")
    hof_fit = argschema.InputFile(description="")
    hof = argschema.InputFile(description="")


class PopulationSelectionPaths(mm.Schema):
    fits = mm.fields.Nested(ModelFit, description="", many=True)


class PopulationSelectionParameters(argschema.ArgSchema):
    paths = mm.fields.Nested(PopulationSelectionPaths)


def main():
    module = argschema.ArgSchemaParser(schema_type=PopulationSelectionParameters)

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
