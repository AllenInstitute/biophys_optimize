Getting started
===============

The `biophys_optimize` package enables the systematic generation of biophysical models based on a standard stimulus set. It has been built to expect data in the form of `NWB files <http://https://www.nwb.org>`_, but it could be customized to use other data formats.

Model generation is broken into several stages to allow for easier workflow management. These are executed by different `scripts <source/reference/scripts>` that draw upon library functions. The scripts use JSON files as inputs and outputs to pass information from one stage to another. Individual scripts could be customized or replaced depending on your needs. This guide will cover the typical stages of fitting a model.

Preprocessing
-------------

The preprocessing stage performs several tasks before model fitting begins. Its tasks include:

* Comparing "Core 1" to "Core 2" sweeps and selecting one as a training sweep
* Calculating target features from the training sweep
* Identifying the maximum current used to perform depolarization block checks during optimization
* Averaging and saving the traces for passive parameter fitting
* Selecting the "fit style" to use for optimization

To do this, the :mod:`~biophys_optimize.scripts.run_passive_fitting` script expects a number of parameters in its input JSON file. An example of that input file looks like:

    {
        "paths": {
            "nwb": "/path/to/nwb/file.nwb",
            "swc": "/path/to/swc/file.swc",
            "storage_directory": "/path/to/fitting_storage_directory"
        },
        "dendrite_type": "spiny",
        "sweeps": {
            "core_1_long_squares": [31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45],
            "core_2_long_squares": [63, 64, 65, 67, 68, 69, 71, 73, 75, 76, 77],
            "seed_1_noise": [52, 56, 58],
            "seed_2_noise": [53, 55, 57, 59],
            "cap_checks": [61, 62]
        },
        "bridge_avg": ,
        "junction_potential": -14.0,
    }