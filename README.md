# biophys_optimize
Optimization of single-cell biophysically detailed models

Installation:
```bash
$ python setup.py install
```

# 1) Pre-process

Run:
```bash
$ python -m biophys_optimize.scripts.run_preprocessing --help
```
to get command line options. There are a handful of JSON files for examples in test_input_files.

## a) Download data using AllenSDK and edit JSON file
Follow the Jupyter Notebook at https://github.com/latimerb/GeneralTutorials/blob/master/AllenSDK/cell_types.ipynb to download electrophysiology and SWC data. The specimen ID needs to be added to the test_input_files/test_preprocess_input.json file and you need to update the paths to the data. The sweep IDs are unique for each specimen so you may need to look these up.
      
## b) Run test_preprocessing.py  
```bash
$ python -m biophys_optimize.scripts.run_preprocessing --input_json ./test_input_files/test_preprocess_input.json
```
# 2) Passive fitting

## a) Edit the passive JSON file
Change all the paths in the /test_input_files/test_passive_input.json file just as in step 1a. Other parameters should still be the same. Then run:
```bash
$ python -m biophys_optimize.scripts.run_passive_fitting --input_json ./test_input_files/test_passive_input.json
```
