from builtins import object
import logging
from neuron import h


class NeuronEnvironment(object):
    _log = logging.getLogger(__name__)

    def __init__(self, hoc_files_to_load, mod_library_path):
        self.h = h
        if mod_library_path:
            self.h.nrn_load_dll(mod_library_path.encode('ascii', 'ignore'))
        for file in hoc_files_to_load:
            self.h.load_file(file.encode('ascii', 'ignore'))

    def activate_variable_time_step(self):
        self.h.cvode_active(1)
        self.h.cvode.atolscale("cai", 1e-4)
        self.h.cvode.maxstep(10)

    def deactivate_variable_time_step(self):
        self.h.cvode_active(0)

    def set_temperature(self, celsius_temperature):
        self.h.celsius = celsius_temperature
