# biophys_optimize
Optimization of single-cell biophysically detailed models

# 1) Pre-process

Run:
python -m biophys_optimize.run_preprocessing --help
to get command line options. There are a handful of JSON files for examples in test_input_files.

## a) Download data using AllenSDK
      Follow the Jupyter Notebook at https://github.com/latimerb/GeneralTutorials/blob/master/AllenSDK/cell_types.ipynb to download electrophysiology and SWC data. The specimen ID needs to be added to the test_preprocess_input.json file and you need to update the paths to the data. The sweep IDs are unique for each specimen so you may need to look these up.
      
## b) Run test_preprocessing.py  
python -m biophys_optimize.scripts.run_preprocessing --input_json ./test_input_files/test_preprocess_input.json

