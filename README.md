# biophys_optimize
Optimization of single-cell biophysically detailed models

## NWB1 and NWB2 data files

We are transitioning from supporting electrophysiology data files in the NWB1 format to the NWB2 format.
[New data](https://portal.brain-map.org/explore/classes/multimodal-characterization) has been released
to the DANDI archive in the newer NWB2 format. We have released IPFX v1.0 to support that data format, which is available on PyPI.

However, the data files in the [Allen Cell Types Database](http://celltypes.brain-map.org) are still in NWB1 format at this time. IPFX 1.0 does not
support NWB1 files. Therefore, to work with those older files, you must install an earlier version of IPFX, which is not available on PyPI (v.1.0 was the first
IPFX version to be deployed via PyPI).

This version of `biphys_optimize` is intended to work with the Allen Cell Types Database NWB1 files. We are updating `biophys_optimize` to work with NWB2 and IPFX v1.0
in a [new branch](https://github.com/AllenInstitute/biophys_optimize/tree/ipfx_1.0_changes). Eventually, this will be the main branch for biophys_optimize as well; we
plan to make that switch when the files for the cells in the Allen Cell Types Database are available in NWB2 format.


## Installation

You will need the [IPFX](https://github.com/alleninstitute/ipfx) package to perform feature extraction in this package. Specifically, you will need an older version of IPFX that supports NWB1 files, which is not available via PyPI.

If you do not yet have the IPFX repository cloned, do the following:
```bash
$ git clone --branch=nwb1-support https://github.com/AllenInstitute/ipfx.git
$ cd ipfx
$ pip install -e .
```

If you already have a local IPFX repository, do the following in that repository to switch to an earlier version with NWB1 support:
```bash
$ git checkout tags/nwb1-support nwb1-support
```

In either case, you can then install `biophys_optimize` by:
```bash
$ git clone https://github.com/AllenInstitute/biophys_optimize.git
$ cd biophys_optimize
$ pip install -e .
```


## 1) Pre-process

Run:
```bash
$ python -m biophys_optimize.scripts.run_preprocessing --help
```
to get command line options. There are a handful of JSON files for examples in test_input_files.

###a) Download data using AllenSDK and edit JSON file
Follow the Jupyter Notebook at https://github.com/latimerb/GeneralTutorials/blob/master/AllenSDK/cell_types.ipynb to download electrophysiology and SWC data. The specimen ID needs to be added to the test_input_files/test_preprocess_input.json file and you need to update the paths to the data. The sweep IDs are unique for each specimen so you may need to look these up.

### b) Run test_preprocessing.py
```bash
$ python -m biophys_optimize.scripts.run_preprocessing --input_json ./test_input_files/test_preprocess_input.json
```
## 2) Passive fitting

### a) Edit the passive JSON file
Change all the paths in the /test_input_files/test_passive_input.json file just as in step 1a. Other parameters should still be the same. Then run:
```bash
$ python -m biophys_optimize.scripts.run_passive_fitting --input_json ./test_input_files/test_passive_input_1.json
```
