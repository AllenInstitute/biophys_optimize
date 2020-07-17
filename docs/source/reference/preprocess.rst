=============
Preprocessing
=============
	
Sweep selection
===============
.. autosummary::
	:toctree: generated/

	~biophys_optimize.preprocess.select_core_1_or_core_2_sweeps
	~biophys_optimize.preprocess.select_core1_trace
	~biophys_optimize.preprocess.is_sweep_good_quality
	~biophys_optimize.preprocess.has_pause
	~biophys_optimize.check_fi_shift.estimate_fi_shift
	~biophys_optimize.check_fi_shift.fi_shift
	~biophys_optimize.check_fi_shift.fi_curve


Feature analysis
================
.. autosummary::
	:toctree: generated/

	~biophys_optimize.preprocess.target_features
	~biophys_optimize.step_analysis.StepAnalysis


Passive fit preparation
=======================
.. autosummary::
	:toctree: generated/

	~biophys_optimize.preprocess.cap_check_grand_averages
	~biophys_optimize.preprocess.save_grand_averages
	~biophys_optimize.preprocess.passive_fit_window
	
Additional characteristics
==========================
.. autosummary::
	:toctree: generated/
	
	~biophys_optimize.preprocess.max_i_for_depol_block_check
	~biophys_optimize.preprocess.swc_has_apical_compartments
