============
Optimization
============


Evolutionary algorithm optimization
===================================

.. autosummary::
	:toctree: generated/

	~biophys_optimize.optimize.optimize
	~biophys_optimize.optimize.initPopulation
	~biophys_optimize.optimize.eval_param_set
	~biophys_optimize.optimize.uniform
	~biophys_optimize.optimize.best_sum
	~biophys_optimize.optimize.check_for_block
	
	
Post-optimization model selection
=================================

.. autosummary::
	:toctree: generated/
	
	~biophys_optimize.model_selection.select_model
	~biophys_optimize.model_selection.build_fit_data
	~biophys_optimize.model_selection.has_noise_block
	~biophys_optimize.model_selection.fit_info


Starting optimization from existing population
==============================================

.. autosummary::
	:toctree: generated/

	~biophys_optimize.population_selection.population_info
	~biophys_optimize.population_selection.select_starting_population
	

Configuration
=============

.. autosummary::
	:toctree: generated/
	
	~biophys_optimize.utils.Utils
	~biophys_optimize.environment.NeuronEnvironment
	~biophys_optimize.optimize.StimParams


NEURON parallel execution
=========================

.. autosummary::
	:toctree: generated/
	
	~biophys_optimize.neuron_parallel.map
	
