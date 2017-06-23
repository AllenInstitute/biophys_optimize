from neuron import h

_pc = h.ParallelContext()

def map(func, *iterables):
    start_time = pc_time()
    userids = []
    userid = 200 # arbitrary, but needs to be a positive integer
    for args in zip(*iterables):
        args2 = (list(a) for a in args)
        _pc.submit(userid, func, *args2)
        userids.append(userid)
        userid += 1

    results = dict(working())

    end_time = pc_time()
    print "Map took ", end_time - start_time
    return [results[userid] for userid in userids]


def working():
    while _pc.working():
        userid = int(_pc.userid())
        ret = _pc.pyret()
        yield userid, ret


def runworker():
    _pc.runworker()


def done():
    _pc.done()


def pc_time():
    return _pc.time()