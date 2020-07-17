from builtins import zip
from neuron import h
import logging

_pc = h.ParallelContext()

def map(func, *iterables):
    """Submit function (`func`) and arguments (`iterables`) to NEURON parallel context"""
    start_time = pc_time()
    userids = []
    userid = 200 # arbitrary, but needs to be a positive integer
    for args in zip(*iterables):
        args2 = (list(a) for a in args)
        _pc.submit(userid, func, *args2)
        userids.append(userid)
        userid += 1

    results = dict(_working())

    end_time = _pc_time()
    logging.info("Map took {} seconds".format(end_time - start_time))
    return [results[userid] for userid in userids]


def _working():
    while _pc.working():
        userid = int(_pc.userid())
        ret = _pc.pyret()
        yield userid, ret


def _runworker():
    _pc.runworker()


def _done():
    _pc.done()


def _pc_time():
    return _pc.time()