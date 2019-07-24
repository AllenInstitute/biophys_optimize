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
        extra_sweep_features = ["stim_amp"]
        self.analyze_basic_features(sweep_set,
            extra_sweep_features=extra_sweep_features,
            exclude_clipped=exclude_clipped)

    def spikes_data(self):
        """ Return a list of spike feature dataframes"""
        return self._spikes_set

    def sweep_features(self):
        """ Return a sweep feature dataframe"""
        return self._sweep_features