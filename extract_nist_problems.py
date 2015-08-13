"""
A small compat shim between NIST StRD datasets/lmfit models and NM benchmark
classes.
"""
from __future__ import division, print_function

import numpy as np

from NISTModels import Models as nist_models, ReadNistData as read_nist_data
from lsq_problems import LSQBenchmarkProblem, LSQBenchmarkProblemFactory

def collect_nist_problems():
    """
    """
    nist_problems = []

    for mod_name in nist_models:
        dct = read_nist_data(mod_name)
        fcn, _, _ = nist_models[mod_name]
        m = len(dct['x'])
        n = dct['nparams']

        for s in ['start1', 'start2']:
            prblm = LSQBenchmarkProblem(mod_name + '_' + s,
                n, m,   # XXX: n, m
                fun=lambda b: fcn(b, dct['x'], dct['y']),
                jac=None,
                x0=dct[s],
                bounds=(-np.inf, np.inf),
                sparsity=None)
            prblm.nist_dct = dct
            nist_problems.append(prblm)

    return nist_problems

if __name__ == "__main__":
    print([_.name for _ in collect_nist_problems()])
