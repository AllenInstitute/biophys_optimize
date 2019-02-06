import numpy as np


def population_info(fits):
    info = []
    for fit in fits:
        fit_type = fit["fit_type"]
        hof_fit = np.loadtxt(fit["hof_fit"])
        info.append({
            "fit_type": fit_type,
            "best_err": hof_fit.min(),
            "hof": fit["hof"],
        })
    return info


def select_starting_population(populations):
    selection = {}
    for pop in populations:
        ft = pop["fit_type"]
        if ft not in selection or pop["best_err"] < selection[ft]["err"]:
            selection[ft] = {
                "err": pop["best_err"],
                "hof": pop["hof"],
            }
    return selection

