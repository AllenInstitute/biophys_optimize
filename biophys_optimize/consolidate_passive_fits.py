#!/usr/bin/env python

import json
import argparse
import logging
import os.path


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def compare_runs(fit_1, fit_2, fit_3, is_spiny):
    """ Choose a particular fitting outcome by comparing the results of three options

    Generally, the fit with the lowest error is used, but the fit without an electrode (`fit_1`)
    could be used even if the absolute error is higher if the cm results are within 10% of
    the fit with an electrode modeled (`fit_3`).

    If the neuron is spiny (`is_spiny` == True), then the calculation assumes that the soma has
    a Cm = 1 uF/cm2 and that the dendrites have an effectively higher Cm (due to the
    presence of dendritic spines, which are not explicitly modeled)

    Parameters
    ----------
    fit_1, fit_2, fit_3 : :class:`PassiveFitResults`
        Passive fitting results from a fit of all three passive parameters (`fit_1`),
        a fit witih a fixed Ra = 100 Ohm-cm (`fit_2`), and a fit with a simulated
        electrode (`fit_3`)
    is_spiny : bool
        Whether the neuron has dendritic spines (and a correspondingly higher effective Cm
        in the dendrites)

    Returns
    -------
    ra : float
        Axial resistivity
    cm1 : float
        Specific membrane capacitance of somatic / axonal compartments. Set to 1.0 if
        `is_spiny` is True.
    cm2 : float
        Effective specific membrane capacitance of dendritic compartments. If `is_spiny` is
        false, `cm1` will equal `cm2`.
    """
    # Check for various fitting outcomes and pick best results
    cm_rel_delta = (fit_1.cm - fit_3.cm) / fit_1.cm
    if fit_2.err < fit_1.err:
        logger.info("Fixed Ri gave better results than original")
        if fit_2.err < fit_3.err:
            logger.info("Using fixed Ri results")
            fit_for_next_step = fit_2
        else:
            logger.info("Using electrode results")
            fit_for_next_step = fit_3
    elif abs(cm_rel_delta) > 0.1:
        logger.info("Original and electrode fits not in agreement")
        logger.debug("original Cm: %g", fit_1.cm)
        logger.debug("w/ electrode Cm: %g", fit_3.cm)
        if fit_1.err < fit_3.err:
            logger.info("Original has lower error")
            fit_for_next_step = fit_1
        else:
            logger.info("Electrode has lower error")
            fit_for_next_step = fit_3
    else:
        fit_for_next_step = fit_1

    ra = fit_for_next_step.ra
    if is_spiny:
        combo_cm = fit_for_next_step.cm
        a1 = fit_for_next_step.a1
        a2 = fit_for_next_step.a2
        cm1 = 1.0
        cm2 = (combo_cm * (a1 + a2) - a1) / a2
    else:
        cm1 = fit_for_next_step.cm
        cm2 = fit_for_next_step.cm

    return ra, cm1, cm2


