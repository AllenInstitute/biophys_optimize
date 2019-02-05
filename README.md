# biophys_optimize
Optimization of single-cell biophysically detailed models

# 1) Pre-process

Run:
python -m biophys_optimize.run_preprocessing --help
to get command line options. There are a handful of JSON files for examples in test_input_files.

This uses the paths and variables in the test_preprocess_input.json file. Since we don't have the data downloaded right now, it gives an error.

python -m biophys_optimize.scripts.run_preprocessing --input_json ./test_input_files/test_preprocess_input.json

