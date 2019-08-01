import numpy as np
from ipfx import stim_features as stf
from ipfx import feature_extractor as fx
from ipfx.stimulus_protocol_analysis import StimulusProtocolAnalysis

class StepAnalysis(StimulusProtocolAnalysis):
    """ Analysis of responses to step current stimuluation

    Parameters
    ----------
    start: float
        Start time of stimulus interval (seconds)

    end: float
        End time of stimulus interval (seconds)
    """
    def __init__(self, start, end):
        spx = fx.SpikeFeatureExtractor(start=start, end=end)
        sptx = fx.SpikeTrainFeatureExtractor(start, end,
            stim_amp_fn=stf._step_stim_amp)
        super(StepAnalysis, self).__init__(spx, sptx)

    def analyze(self, sweep_set, exclude_clipped=False):
        """ Analyze spike and sweep features

        Parameters
        ----------
        sweep_set: SweepSet

        exclude_clipped: bool (optional, default=False)
            Whether to exclude clipped spikes from sweep-level features
        """
        extra_sweep_features = ["stim_amp", "v_baseline"]
        self.analyze_basic_features(sweep_set,
            extra_sweep_features=extra_sweep_features,
            exclude_clipped=exclude_clipped)

        # Analyze additional spike-level features
        for sd in self._spikes_set:
            if sd.shape[0] >= 2:
                sd["slow_trough_delta_v"] = _slow_trough_delta_v(
                    sd["fast_trough_v"].values, sd["slow_trough_v"].values)
                sd["slow_trough_norm_time"] = _slow_trough_norm_t(
                    sd["threshold_t"].values,
                    sd["slow_trough_t"].values,
                    sd["trough_t"].values)

    def spikes_data(self):
        """ Return a list of spike feature dataframes"""
        return self._spikes_set

    def sweep_features(self):
        """ Return a sweep feature dataframe"""
        return self._sweep_features


def _slow_trough_delta_v(fast_trough_v, slow_trough_v):
    delta = fast_trough_v - slow_trough_v
    delta[np.isnan(delta)] = 0
    return delta


def _slow_trough_norm_t(threshold_t, slow_trough_t, trough_t):
    trough_values = slow_trough_t
    mask = np.isnan(trough_values)
    trough_values[mask] = trough_t[mask]

    isis = np.diff(threshold_t)
    slow_trough_norm_t = np.zeros_like(threshold_t) * np.nan
    slow_trough_norm_t[:-1] = (trough_values[:-1] - threshold_t[:-1]) / isis
    return slow_trough_norm_t

