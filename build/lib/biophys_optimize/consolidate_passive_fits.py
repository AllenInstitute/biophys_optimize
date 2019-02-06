#!/usr/bin/env python

import json
import argparse
import logging
import os.path


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




def compare_runs(preprocess_results, fit_1, fit_2, fit_3):
    # Check for various fitting outcomes and pick best results
    cm_rel_delta = (fit_1["cm"] - fit_3["cm"]) / fit_1["cm"]
    if fit_2["err"] < fit_1["err"]:
        logger.info("Fixed Ri gave better results than original")
        if fit_2["err"] < fit_3["err"]:
            logger.info("Using fixed Ri results")
            fit_for_next_step = fit_2
        else:
            logger.info("Using electrode results")
            fit_for_next_step = fit_3
    elif abs(cm_rel_delta) > 0.1:
        logger.info("Original and electrode fits not in agreement")
        logger.debug("original Cm: %g", fit_1["cm"])
        logger.debug("w/ electrode Cm: %g", fit_3["cm"])
        if fit_1["err"] < fit_3["err"]:
            logger.info("Original has lower error")
            fit_for_next_step = fit_1
        else:
            logger.info("Electrode has lower error")
            fit_for_next_step = fit_3
    else:
        fit_for_next_step = fit_1

    ra = fit_for_next_step["ra"]
    is_spiny = preprocess_results["is_spiny"]
    if is_spiny:
        combo_cm = fit_for_next_step["cm"]
        a1 = fit_for_next_step["a1"]
        a2 = fit_for_next_step["a2"]
        cm1 = 1.0
        cm2 = (combo_cm * (a1 + a2) - a1) / a2
    else:
        cm1 = fit_for_next_step["cm"]
        cm2 = fit_for_next_step["cm"]

    return ra, cm1, cm2


