import numpy as np


def population_info(fits):
    """ Construct information about fit populations

    Parameters
    ----------
    fits : list
        List of dictionaries with `fit_type`, hall-of-fame file path (`hof`), and
        hall-of-fame error values file path (`hof_fit`)

    Returns
    -------
    info : list
        List of dictionaries summarizing the fit type (`fit_type`), best error encountered
        in hall-of-fame population (`best_err`), and file path of hall-of-fame population
        (`hof`).
    """
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
    """ Choose starting population for next step from different populations

    Typically different populations come from different random seed initializations

    Parameters
    ----------
    populations : list
        List of dictionaries summarizing the fit type (`fit_type`), best error encountered
        in hall-of-fame population (`best_err`), and file path of hall-of-fame population

    Returns
    -------
    selection : dict
        Dictionary with different fit types as keys and best populations for fit types
        as values
    """
    selection = {}
    for pop in populations:
        ft = pop["fit_type"]
        if ft not in selection or pop["best_err"] < selection[ft]["err"]:
            selection[ft] = {
                "err": pop["best_err"],
                "hof": pop["hof"],
            }
    return selection

