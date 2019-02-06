import numpy as np
import logging
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import sweep_functions as sf
from neuron import h

class Utils:
    _log = logging.getLogger(__name__)

    def __init__(self, hoc_files_to_load, mod_library_path):
        self.h = h
        if mod_library_path:
            self.h.nrn_load_dll(mod_library_path.encode('ascii', 'ignore'))
        for file in hoc_files_to_load:
            self.h.load_file(file.encode('ascii', 'ignore'))
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None
        self.cell = self.h.cell()

    def generate_morphology(self, morph_filename):
        cell = self.cell

        swc = self.h.Import3d_SWC_read()
        swc.quiet = 1
        swc.input(morph_filename.encode('ascii', 'ignore'))
        imprt = self.h.Import3d_GUI(swc, 0)
        imprt.instantiate(cell)

        for seg in cell.soma[0]:
            seg.area()

        for sec in cell.all:
            sec.nseg = 1 + 2 * int(sec.L / 40)

        cell.simplify_axon()
        for sec in cell.axonal:
            sec.L = 30
            sec.diam = 1
            sec.nseg = 1 + 2 * int(sec.L / 40)
        cell.axon[0].connect(cell.soma[0], 0.5, 0)
        cell.axon[1].connect(cell.axon[0], 1, 0)
        h.define_shape()

    def load_cell_parameters(self, passive, conditions, channels, addl_params):
        cell = self.cell
        self.channels = channels
        self.addl_params = addl_params

        # Set passive properties
        for sec in cell.all:
            sec.Ra = passive['ra']
            sec.cm = passive['cm'][sec.name().split(".")[1][:4]]
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = passive["e_pas"]
        self.h.v_init = passive["e_pas"]

        # Insert channels and set parameters
        for c in channels:
            if c["mechanism"] != "":
                sections = [s for s in cell.all if s.name().split(".")[1][:4] == c["section"]]
                for sec in sections:
                    sec.insert(c["mechanism"])

        for ap in addl_params:
            if ap["mechanism"] != "":
                sections = [s for s in cell.all if s.name().split(".")[1][:4] == ap["section"]]
                for sec in sections:
                    sec.insert(ap["mechanism"])

        # Set reversal potentials
        for erev in conditions['erev']:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == erev["section"]]
            for sec in sections:
                sec.ena = erev["ena"]
                sec.ek = erev["ek"]

    def set_normalized_parameters(self, params):
        channels_and_others = self.channels + self.addl_params
        for i, p in enumerate(params):
            c = channels_and_others[i]
            if p > 1.0 or p < 0.0:
                print "WARNING: Setting a normalized parameter with a value outside [0, 1] ({:s} set to {:f})".format(c, p)
            value = p * (c["max"] - c["min"]) + c["min"]
            sections = [s for s in self.cell.all if s.name().split(".")[1][:4] == c["section"]]
            for sec in sections:
                param_name = c["parameter"]
                if c["mechanism"] != "":
                    param_name += "_" + c["mechanism"]
                setattr(sec, param_name, value)

    def set_actual_parameters(self, params):
        channels_and_others = self.channels + self.addl_params
        for i, p in enumerate(params):
            c = channels_and_others[i]
            sections = [s for s in self.cell.all if s.name().split(".")[1][:4] == c["section"]]
            for sec in sections:
                param_name = c["parameter"]
                if c["mechanism"] != "":
                    param_name += "_" + c["mechanism"]
                setattr(sec, param_name, p)

    def normalize_actual_parameters(self, params):
        params_array = np.array(params)
        channels_and_others = self.channels + self.addl_params
        max_vals = np.array([c["max"] for c in channels_and_others])
        min_vals = np.array([c["min"] for c in channels_and_others])

        normalized_params = (params_array - min_vals) / (max_vals - min_vals)
        return normalized_params.tolist()

    def actual_parameters_from_normalized(self, params):
        actual_params = []
        channels_and_others = self.channels + self.addl_params
        for i, p in enumerate(params):
            c = channels_and_others[i]
            value = p * (c["max"] - c["min"]) + c["min"]
            actual_params.append(value)
        return actual_params

    def insert_iclamp(self):
        self.stim = self.h.IClamp(self.cell.soma[0](0.5))

    def set_iclamp_params(self, amp, delay, dur):
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def calculate_feature_errors(self, t_ms, v, i, feature_names, targets):
        # Special case checks and penalty values
        minimum_num_spikes = 2
        missing_penalty_value = 20.0
        max_fail_penalty = 250.0
        min_fail_penalty = 75.0
        overkill_reduction = 0.75
        variance_factor = 0.1

        fail_trace = False

        delay = self.stim.delay * 1e-3
        duration = self.stim.dur * 1e-3
        t = t_ms * 1e-3

        # penalize for failing to return to rest
        start_index = np.flatnonzero(t >= delay)[0]
        if np.abs(v[-1] - v[:start_index].mean()) > 2.0:
            fail_trace = True
        else:
            pre_swp = EphysSweepFeatureExtractor(t, v, i, start=0, end=delay)
            pre_swp.process_spikes()
            if pre_swp.sweep_feature("avg_rate") > 0:
                fail_trace = True

        target_features_dict = {f["name"]: {"mean": f["mean"], "stdev": f["stdev"]} for f in targets}

        if not fail_trace:
            swp = EphysSweepFeatureExtractor(t, v, i, start=delay, end=delay + duration)
            swp.process_spikes()
            swp.sweep_feature("v_baseline") # Pre-compute the baseline
            swp.process_new_spike_feature("slow_trough_norm_t",
                                          sf.slow_trough_norm_t,
                                          affected_by_clipping=True)
            swp.process_new_spike_feature("slow_trough_delta_v",
                                          sf.slow_trough_delta_voltage_feature,
                                          affected_by_clipping=True)
            sweep_keys = swp.sweep_feature_keys()
            spike_keys = swp.spike_feature_keys()

            if len(swp.spike_feature("threshold_t")) < minimum_num_spikes: # Enough spikes?
                fail_trace = True
            else:
                avg_per_spike_peak_error = np.mean([abs(peak_v - target_features_dict["peak_v"]["mean"])
                                                    for peak_v in swp.spike_feature("peak_v")])
                avg_overall_error = abs(target_features_dict["peak_v"]["mean"] -
                                        swp.spike_feature("peak_v").mean())
                if avg_per_spike_peak_error > 3. * avg_overall_error: # Weird bi-modality of spikes; 3.0 is arbitrary
                    fail_trace = True

        if fail_trace:
            variance_start = np.flatnonzero(t >= delay - 0.1)[0]
            variance_end = np.flatnonzero(t >= (delay + duration) / 2.0)[0]
            trace_variance = v[variance_start:variance_end].var()
            error_value = max(max_fail_penalty - trace_variance * variance_factor, min_fail_penalty)
            errs = np.ones(len(feature_names)) * error_value
        else:
            errs = []
            for k in feature_names:
                if k in sweep_keys:
                    model_mean = swp.sweep_feature(k)
                elif k in spike_keys:
                    model_mean = swp.spike_feature(k).mean()
                else:
                    _log.debug("Could not find feature %s", k)
                    errs.append(missing_penalty_value)
                    continue
                if np.isnan(model_mean):
                    errs.append(missing_penalty_value)
                else:
                    target_mean = target_features_dict[k]['mean']
                    target_stdev = target_features_dict[k]['stdev']
                    errs.append(np.abs((model_mean - target_mean) / target_stdev))
            errs = np.array(errs)
        return errs

    def calculate_features(self, t_ms, v, i, feature_names):
        delay = self.stim.delay * 1e-3
        duration = self.stim.dur * 1e-3
        t = t_ms * 1e-3

        swp = EphysSweepFeatureExtractor(t, v, i, start=delay, end=delay + duration)
        swp.process_spikes()
        swp.sweep_feature("v_baseline") # Pre-compute the baseline
        swp.process_new_spike_feature("slow_trough_norm_t",
                                      sf.slow_trough_norm_t,
                                      affected_by_clipping=True)
        swp.process_new_spike_feature("slow_trough_delta_v",
                                      sf.slow_trough_delta_voltage_feature,
                                      affected_by_clipping=True)
        sweep_keys = swp.sweep_feature_keys()
        spike_keys = swp.spike_feature_keys()
        out_features = []
        for k in feature_names:
            if k in sweep_keys:
                out_features.append(swp.sweep_feature(k))
            elif k in spike_keys:
                out_features.append(swp.spike_feature(k).mean())
            else:
                out_features.append(np.nan)
        return np.array(out_features)

    def record_values(self):
        v_vec = self.h.Vector()
        t_vec = self.h.Vector()
        i_vec = self.h.Vector()

        v_vec.record(self.cell.soma[0](0.5)._ref_v)
        i_vec.record(self.stim._ref_amp)
        t_vec.record(self.h._ref_t)

        return v_vec, i_vec, t_vec
