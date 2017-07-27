#!/usr/bin/bash

export ALLENSDK_PATH=/data/informatics/mousecelltypes/biophys_optimize/allensdk/
export ARGSCHEMA_PATH=/data/informatics/mousecelltypes/biophys_optimize/argschema/
export MM_PATH=/data/informatics/mousecelltypes/biophys_optimize/marshmallow/

NEURON_HOME=/shared/utils.x86_64/nrn-7.4-1370
export PYTHONPATH=${NEURON_HOME}/lib/python:${ALLENSDK_PATH}:${ARGSCHEMA_PATH}:${MM_PATH}:${PYTHONPATH}
export PYTHON_HOME=/shared/utils.x86_64/python-2.7
export PYTHON=${PYTHON_HOME}/bin/python
export PATH=${PYTHON_HOME}/bin:${NEURON_HOME}/x86_64/bin:${PATH}
NUM_PROCS=240

mpiexec -np $NUM_PROCS $PYTHON $@
