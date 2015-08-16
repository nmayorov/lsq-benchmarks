"""Benchmark problems for nonlinear least squares."""


from __future__ import division, print_function

import inspect
import os
import sys

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.sparse import lil_matrix
from scipy.optimize._numdiff import check_derivative, group_columns
from scipy.optimize._lsq.common import make_strictly_feasible


class LSQBenchmarkProblem(object):
    """Class describing nonlinear least squares problem.

    The optimized variable is n-dimensional vector x and the objective function
    has the form

    F(x) = ||f(x)||^2 = sum(f_i(x)^2, i = 1, ..., m)

    Where f is a vector function f = (f_1, ..., f_m), we call f_i as residuals.

    Jacobian of f is an m by n matrix, its (i, j) element is the partial
    derivative of f_i with respect to x_j.

    Parameters
    ----------
    n : int
        Number of independent variable.s
    m : int
        Number of residuals.
    fun : callable
        Function returning ndarray with m residuals.
    jac : None or callable, optional
        Function returning Jacobian matrix as ndarray of shape (m, n).
        If None use finite difference derivative estimation.
    x0 : ndarray, shape(n,)
        Initial guess for optimized variable.
    bounds : tuple of array or None, optional
        Lower and upper bounds on independent variables.
    """

    def __init__(self, name, n, m, fun, jac, x0, bounds=(-np.inf, np.inf),
                 sparsity=None):
        self.name = name
        self.n = n
        self.m = m
        self.fun = fun
        self.x0 = x0
        self.jac = jac
        self.bounds = bounds
        self.sparsity = sparsity

    def obj_value(self, x):
        f = self.fun(x)
        return 0.5 * np.dot(f, f)

    def grad(self, x):
        f = self.fun(x)
        J = self.jac(x)
        return J.T.dot(f)

    def check_jacobian(self):
        if self.jac is None:
            return None

        if self.sparsity is not None:
            sparse_diff = True
            groups = group_columns(self.sparsity)
            sparsity = (self.sparsity, groups)
        else:
            sparse_diff = False
            sparsity = None

        lb, ub = [np.resize(b, self.n) for b in self.bounds]
        x = self.x0 + np.random.randn(self.n)
        x = make_strictly_feasible(x, lb, ub)
        return check_derivative(self.fun, self.jac, x, bounds=self.bounds,
                                sparse_diff=sparse_diff, sparsity=sparsity)


class LSQBenchmarkProblemFactory(object):
    """Class for creating least-squares problems with different bounds
    and starting points."""
    def __init__(self, n, m, specs, sparsity=None):
        self.n = n
        self.m = m
        self.specs = specs
        self.sparsity = sparsity

    @staticmethod
    def _adjust_names(problems, suffix=""):
        for i, p in enumerate(problems):
            p.name += suffix
            if len(problems) > 1:
                p.name += "_" + str(i)

    def extract_problems(self):
        unbounded = []
        bounded = []
        name = self.__class__.__name__
        for x0, bounds in self.specs:
            x0 = np.asarray(x0, dtype=float)
            problem = LSQBenchmarkProblem(
                name, self.n, self.m, self.fun, self.jac, x0, bounds=bounds,
                sparsity=self.sparsity)

            lb, ub = bounds
            if np.any(lb != -np.inf) or (ub != np.inf):
                bounded.append(problem)
            else:
                unbounded.append(problem)

        self._adjust_names(unbounded)
        self._adjust_names(bounded, "_B")

        return unbounded, bounded

    def fun(self, x):
        raise NotImplementedError

    def jac(self, x):
        raise NotImplementedError


class CoatingThickness(LSQBenchmarkProblemFactory):
    """Coating thickness standardization problem, [1]_.

    Number of variables --- 134, number of residuals --- 252, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection"
    """

    def __init__(self):
        x0 = np.hstack(([-8.0, 13.0, 1.2, 0.2, 0.1, 6.0, 5.5, -5.2],
                        np.zeros(126)))
        specs = [
            (x0, (-np.inf, np.inf))
        ]
        super(CoatingThickness, self).__init__(134, 252, specs)

        self.n0 = self.m // 4
        self.xi = np.array([
            [0.7140, 0.7169, 0.7232, 0.7151, 0.6848, 0.7070, 0.7177, 0.7073,
             0.6734, 0.7174, 0.7125, 0.6947, 0.7121, 0.7166, 0.6894, 0.6897,
             0.7024, 0.7026, 0.6800, 0.6957, 0.6987, 0.7111, 0.7097, 0.6809,
             0.7139, 0.7046, 0.6950, 0.7032, 0.7019, 0.6975, 0.6955, 0.7056,
             0.6965, 0.6848, 0.6995, 0.6105, 0.6027, 0.6084, 0.6081, 0.6057,
             0.6116, 0.6052, 0.6136, 0.6032, 0.6081, 0.6092, 0.6122, 0.6157,
             0.6191, 0.6169, 0.5483, 0.5371, 0.5576, 0.5521, 0.5495, 0.5499,
             0.4937, 0.5092, 0.5433, 0.5018, 0.5363, 0.4977, 0.5296],
            [5.145, 5.241, 5.389, 5.211, 5.154, 5.105, 5.191, 5.013, 5.582,
             5.208, 5.142, 5.284, 5.262, 6.838, 6.215, 6.817, 6.889, 6.732,
             6.717, 6.468, 6.776, 6.574, 6.465, 6.090, 6.350, 4.255, 4.154,
             4.211, 4.287, 4.104, 4.007, 4.261, 4.150, 4.040, 4.155, 5.086,
             5.021, 5.040, 5.247, 5.125, 5.136, 4.949, 5.253, 5.154, 5.227,
             5.120, 5.291, 5.294, 5.304, 5.209, 5.384, 5.490, 5.563, 5.532,
             5.372, 5.423, 7.237, 6.944, 6.957, 7.138, 7.009, 7.074, 7.046]
        ])
        self.y = np.array(
            [9.3636, 9.3512, 9.4891, 9.1888, 9.3161, 9.2585, 9.2913, 9.3914,
             9.4524, 9.4995, 9.4179, 9.468, 9.4799, 11.2917, 11.5062, 11.4579,
             11.3977, 11.3688, 11.3897, 11.3104, 11.3882, 11.3629, 11.3149,
             11.2474, 11.2507, 8.1678, 8.1017, 8.3506, 8.3651, 8.2994, 8.1514,
             8.2229, 8.1027, 8.3785, 8.4118, 8.0955, 8.0613, 8.0979, 8.1364,
             8.1700, 8.1684, 8.0885, 8.1839, 8.1478, 8.1827, 8.029, 8.1000,
             8.2579, 8.2248, 8.2540, 6.8518, 6.8547, 6.8831, 6.9137, 6.8984,
             6.8888, 8.5189, 8.5308, 8.5184, 8.5222, 8.5705, 8.5353, 8.5213,
             8.3158, 8.1995, 8.2283, 8.1857, 8.2738, 8.2131, 8.2613, 8.2315,
             8.2078, 8.2996, 8.3026, 8.0995, 8.2990, 9.6753, 9.6687, 9.5704,
             9.5435, 9.6780, 9.7668, 9.7827, 9.7844, 9.7011, 9.8006, 9.7610,
             9.7813, 7.3073, 7.2572, 7.4686, 7.3659, 7.3587, 7.3132, 7.3542,
             7.2339, 7.4375, 7.4022, 10.7914, 10.6554, 10.7359, 10.7583,
             10.7735, 10.7907, 10.6465, 10.6994, 10.7756, 10.7402, 10.6800,
             10.7000, 10.8160, 10.6921, 10.8677, 12.3495, 12.4424, 12.4303,
             12.5086, 12.4513, 12.4625, 16.2290, 16.2781, 16.2082, 16.2715,
             16.2464, 16.1626, 16.1568]
        )

        self.scale1 = 4.08
        self.scale2 = 0.417

    def fun(self, x):
        xi = np.vstack(
            (self.xi[0] + x[8:8 + self.n0],
             self.xi[1] + x[8 + self.n0:])
        )
        z1 = x[0] + x[1] * xi[0] + x[2] * xi[1] + x[3] * xi[0] * xi[1]
        z2 = x[4] + x[5] * xi[0] + x[6] * xi[1] + x[7] * xi[0] * xi[1]
        return np.hstack(
            (z1 - self.y[:self.n0],
             z2 - self.y[self.n0:],
             self.scale1 * x[8:8 + self.n0],
             self.scale2 * x[8 + self.n0:])
        )

    def jac(self, x):
        J = np.zeros((self.m, self.n))
        ind = np.arange(self.n0)
        xi = np.vstack(
            (self.xi[0] + x[8:8 + self.n0],
             self.xi[1] + x[8 + self.n0:])
        )
        J[:self.n0, 0] = 1
        J[:self.n0, 1] = xi[0]
        J[:self.n0, 2] = xi[1]
        J[:self.n0, 3] = xi[0] * xi[1]
        J[ind, ind + 8] = x[1] + x[3] * xi[1]
        J[ind, ind + 8 + self.n0] = x[2] + x[3] * xi[0]

        J[self.n0:2 * self.n0, 4] = 1
        J[self.n0:2 * self.n0, 5] = xi[0]
        J[self.n0:2 * self.n0, 6] = xi[1]
        J[self.n0:2 * self.n0, 7] = xi[0] * xi[1]
        J[ind + self.n0, ind + 8] = x[5] + x[7] * xi[1]
        J[ind + self.n0, ind + 8 + self.n0] = x[6] + x[7] * xi[0]

        J[ind + 2 * self.n0, ind + 8] = self.scale1
        J[ind + 3 * self.n0, ind + 8 + self.n0] = self.scale2

        return J


class ExponentialFitting(LSQBenchmarkProblemFactory):
    """The problem of fitting the sum of exponentials with linear degrees
    to data, [1]_.

    Number of variables --- 5, number of residuals --- 33, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
    """

    def __init__(self):
        x0 = np.array([0.5, 1.5, -1, 1e-2, 2e-2])
        specs = [
            (x0, (-np.inf, np.inf))
        ]

        super(ExponentialFitting, self).__init__(5, 33, specs)
        self.t = np.arange(self.m, dtype=float) * 10
        self.y = 1e-1 * np.array(
            [8.44, 9.08, 9.32, 9.36, 9.25, 9.08, 8.81, 8.5, 8.18,
             7.84, 7.51, 7.18, 6.85, 6.58, 6.28, 6.03, 5.8, 5.58,
             5.38, 5.22, 5.06, 4.9, 4.78, 4.67, 4.57, 4.48, 4.38,
             4.31, 4.24, 4.2, 4.14, 4.11, 4.06]
        )

    def fun(self, x):
        return (x[0] + x[1] * np.exp(-x[3] * self.t) +
                x[2] * np.exp(-x[4] * self.t) - self.y)

    def jac(self, x):
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = np.exp(-x[3] * self.t)
        J[:, 2] = np.exp(-x[4] * self.t)
        J[:, 3] = -x[1] * self.t * np.exp(-x[3] * self.t)
        J[:, 4] = -x[2] * self.t * np.exp(-x[4] * self.t)
        return J


class GaussianFittingI(LSQBenchmarkProblemFactory):
    """The problem of fitting the sum of exponentials with linear and
    quadratic degrees to data, [1]_.

    Number of variables --- 11, number of residuals --- 65, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection"
    """

    def __init__(self):
        x0 = np.array([1.3, 6.5e-1, 6.5e-1, 7.0e-1, 6.0e-1,
                       3.0, 5.0, 7.0, 2.0, 4.5, 5.5])
        specs = [
            (x0, (-np.inf, np.inf))
        ]
        super(GaussianFittingI, self).__init__(11, 65, specs)
        self.t = np.arange(self.m, dtype=float) * 1e-1
        self.y = np.array(
            [1.366, 1.191, 1.112, 1.013, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1,
             7.86e-1, 7.25e-1, 7.46e-1, 6.79e-1, 6.08e-1, 6.55e-1, 6.16e-1,
             6.06e-1, 6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1, 6.49e-1,
             6.94e-1, 6.44e-1, 6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1,
             4.95e-1, 5.0e-1, 4.23e-1, 3.95e-1, 3.75e-1, 3.72e-1, 3.91e-1,
             3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, 5.23e-1, 5.62e-1, 6.07e-1,
             6.53e-1, 6.72e-1, 7.08e-1, 6.33e-1, 6.68e-1, 6.45e-1, 6.32e-1,
             5.91e-1, 5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1, 7.29e-1,
             7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1, 1.62e-1, 9.8e-2,
             5.4e-2]
        )

    def fun(self, x):
        return (x[0] * np.exp(-x[4] * self.t) +
                x[1] * np.exp(-x[5] * (self.t - x[8]) ** 2) +
                x[2] * np.exp(-x[6] * (self.t - x[9]) ** 2) +
                x[3] * np.exp(-x[7] * (self.t - x[10]) ** 2) - self.y)

    def jac(self, x):
        J = np.empty((self.m, self.n))
        e0 = np.exp(-x[4] * self.t)
        e1 = np.exp(-x[5] * (self.t - x[8]) ** 2)
        e2 = np.exp(-x[6] * (self.t - x[9]) ** 2)
        e3 = np.exp(-x[7] * (self.t - x[10]) ** 2)
        J[:, 0] = e0
        J[:, 1] = e1
        J[:, 2] = e2
        J[:, 3] = e3
        J[:, 4] = -x[0] * self.t * e0
        J[:, 5] = -x[1] * (self.t - x[8]) ** 2 * e1
        J[:, 6] = -x[2] * (self.t - x[9]) ** 2 * e2
        J[:, 7] = -x[3] * (self.t - x[10]) ** 2 * e3
        J[:, 8] = 2 * x[1] * x[5] * (self.t - x[8]) * e1
        J[:, 9] = 2 * x[2] * x[6] * (self.t - x[9]) * e2
        J[:, 10] = 2 * x[3] * x[7] * (self.t - x[10]) * e3
        return J


class ThermistorResistance(LSQBenchmarkProblemFactory):
    """The problem of fitting thermistor parameters to data, [1]_.

    Number of variables --- 3, number of residuals --- 16, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
    """

    def __init__(self):
        x0 = np.array([2e-2, 4e3, 2.5e2])
        specs = [
            (x0, (-np.inf, np.inf))
        ]
        super(ThermistorResistance, self).__init__(3, 16, specs)

        self.t = 50.0 + 5 * np.arange(self.m)
        self.y = np.array(
            [3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4,
             9.744e3, 8.261e3, 7.03e3, 6.005e3, 5.147e3, 4.427e3, 3.82e3,
             3.307e3, 2.872e3]
        )

    def fun(self, x):
        return x[0] * np.exp(x[1] / (self.t + x[2])) - self.y

    def jac(self, x):
        J = np.empty((self.m, self.n))
        e = np.exp(x[1] / (self.t + x[2]))
        J[:, 0] = e
        J[:, 1] = x[0] / (self.t + x[2]) * e
        J[:, 2] = -x[0] * x[1] * (self.t + x[2]) ** -2 * e
        return J


class EnzymeReaction(LSQBenchmarkProblemFactory):
    """The problem of fitting kinetic parameters for an enzyme reaction, [1]_.

    Number of variables --- 4, number of residuals --- 11, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
    """

    def __init__(self):
        x0 = np.array([2.5, 3.9, 4.15, 3.9]) * 1e-1
        specs = [
            (x0, (-np.inf, np.inf))
        ]
        super(EnzymeReaction, self).__init__(4, 11, specs)

        self.u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1,
                           1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
        self.y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2,
                           6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2,
                           2.46e-2])

    def fun(self, x):
        return (x[0] * (self.u ** 2 + x[1] * self.u) /
                (self.u ** 2 + x[2] * self.u + x[3]) - self.y)

    def jac(self, x):
        J = np.empty((self.m, self.n))
        den = self.u ** 2 + x[2] * self.u + x[3]
        num = self.u ** 2 + x[1] * self.u
        J[:, 0] = num / den
        J[:, 1] = x[0] * self.u / den
        J[:, 2] = -x[0] * num * self.u / den ** 2
        J[:, 3] = -x[0] * num / den ** 2
        return J


class ChebyshevQuadrature(LSQBenchmarkProblemFactory):
    """The problem of determining the optimal nodes of a quadrature formula
     with equal weights, [1]_.

    Number of variables --- 11, number of residuals --- 11, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 30
    """

    def __init__(self, n):
        self.x0 = (1 + np.arange(n)) / (n + 1)
        specs = []
        super(ChebyshevQuadrature, self).__init__(n, n, specs)

        cp = Chebyshev(1)
        self.T_all = [cp.basis(i + 1, domain=[0.0, 1.0]) for i in range(n)]

    def fun(self, x):
        f = np.empty(self.n)
        for i in range(self.m):
            T = self.T_all[i]
            f[i] = np.mean(T(x)) - T.integ(lbnd=0.0)(1.0)
        return f

    def jac(self, x):
        J = np.empty((self.m, self.n))
        for i in range(self.m):
            T = self.T_all[i]
            J[i] = T.deriv()(x)
        J /= self.n
        return J


class ChebyshevQuadrature7(ChebyshevQuadrature):
    def __init__(self):
        super(ChebyshevQuadrature7, self).__init__(7)
        x0_1 = self.x0.copy()
        x0_1[:3] = np.array([0.025, 0.1, 0.15])
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (x0_1, (np.zeros(self.n), [0.05, 0.23, 0.333, 1, 1, 1, 1]))
        ]


class ChebyshevQuadrature8(ChebyshevQuadrature):
    def __init__(self):
        super(ChebyshevQuadrature8, self).__init__(8)
        x0_1 = self.x0.copy()
        x0_1[:3] = np.array([0.02, 0.1, 0.2])
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (x0_1, ([0, 0, 0.1, 0, 0, 0, 0, 0],
                    [0.04, 0.2, 0.3, 1, 1, 1, 1, 1]))
        ]


class ChebyshevQuadrature9(ChebyshevQuadrature):
    def __init__(self):
        super(ChebyshevQuadrature9, self).__init__(9)
        self.specs += [
            (self.x0, (-np.inf, np.inf))
        ]


class ChebyshevQuadrature10(ChebyshevQuadrature):
    def __init__(self):
        super(ChebyshevQuadrature10, self).__init__(10)
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (self.x0, ([0, 0.1, 0.2, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5],
                       [1, 0.2, 0.3, 0.4, 0.5, 1, 1, 1, 1, 1]))
        ]


class ChebyshevQuadrature11(ChebyshevQuadrature):
    def __init__(self):
        super(ChebyshevQuadrature11, self).__init__(11)
        self.specs += [
            (self.x0, (-np.inf, np.inf))
        ]


class Rosenbrock(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0_1 = np.array([-2.0, 1.0])
        x0_2 = np.array([2.0, 2.0])
        x0_3 = np.array([-2.0, 2.0])
        x0_4 = np.array([0.0, 2.0])
        x0_5 = np.array([-1.2, 1.0])
        specs = [
            (x0_1, (-np.inf, np.inf)),
            (x0_1, ([-np.inf, -1.5], np.inf)),
            (x0_2, ([-np.inf, 1.5], np.inf)),
            (x0_3, ([-np.inf, 1.5], np.inf)),
            (x0_4, ([-np.inf, 1.5], [1.0, np.inf])),
            (x0_2, ([1.0, 1.5], [3.0, 3.0])),
            (x0_5, ([-50.0, 0.0], [0.5, 100]))
        ]
        super(Rosenbrock, self).__init__(2, 2, specs)

    def fun(self, x):
        return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    def jac(self, x):
        return np.array([
            [-20 * x[0], 10],
            [-1, 0]
        ])


class RosenbrockExtended(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        np.random.seed(0)

        if n % 2 != 0:
            raise ValueError("`n` must be even.")

        x0 = np.empty(n)
        x0[::2] = -1.2
        x0[1::2] = 3.0

        x1 = 2.0 * np.ones(n)

        x2 = np.empty(n)
        x2[::2] = -2
        x2[1::2] = 2

        x3 = np.empty(n)
        x3[::2] = 0
        x3[1::2] = 2

        x4 = np.empty(n)
        x4[::2] = -1.2
        x4[1::2] = 1.0

        for x in [x0, x1, x2, x3, x4]:
            x += 0.1 * np.random.rand(n)

        lb1 = np.empty(n)
        lb1[::2] = -np.inf
        lb1[1::2] = -1.5

        lb2 = np.empty(n)
        lb2[::2] = -np.inf
        lb2[1::2] = 1.5

        ub2 = np.empty(n)
        ub2[::2] = 1
        ub2[1::2] = np.inf

        lb3 = np.empty(n)
        lb3[::2] = 1
        lb3[1::2] = 1.5

        ub3 = 3 * np.ones(n)

        lb4 = np.empty(n)
        lb4[::2] = -50
        lb4[1::2] = 0

        ub4 = np.empty(n)
        ub4[::2] = 0.5
        ub4[1::2] = 100

        for b in [lb1, lb2, lb3, lb4, ub2, ub3, ub4]:
            b += 0.1 * np.random.rand(n)

        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, (lb1, np.inf)),
            (x1, (lb2, np.inf)),
            (x2, (lb2, np.inf)),
            (x3, (lb2, ub2)),
            (x1, (lb3, ub3)),
            (x4, (lb4, ub4))
        ]

        sparsity = lil_matrix((n, n), dtype=int)
        i1 = np.arange(0, n, 2)
        i2 = np.arange(1, n, 2)
        sparsity[i1, i1] = 1
        sparsity[i1, i2] = 1
        sparsity[i2, i1] = 1

        super(RosenbrockExtended, self).__init__(n, n, specs,
                                                 sparsity=sparsity)

    def fun(self, x):
        f = np.empty(self.n)
        i = np.arange(0, self.n, 2)
        f[i] = 10 * (x[i + 1] - x[i]**2)
        i = np.arange(1, self.n, 2)
        f[i] = 1 - x[i - 1]
        return f

    def jac(self, x):
        J = lil_matrix((self.n, self.n))
        i1 = np.arange(0, self.n, 2)
        i2 = np.arange(1, self.n, 2)
        J[i1, i1] = -20 * x[i1]
        J[i1, i2] = 10
        J[i2, i1] = -1
        return J


class RosenbrockExtended10K(RosenbrockExtended):
    def __init__(self):
        super(RosenbrockExtended10K, self).__init__(10000)


class BroydenBanded(LSQBenchmarkProblemFactory):
    def __init__(self, n, l, u):
        n = n
        self.l = l
        self.u = u
        self.x0 = -np.ones(n)
        specs = [
            (self.x0, (-np.inf, np.inf))
        ]

        sparsity = lil_matrix((n, n), dtype=int)
        for i in range(n):
            li = max(i - self.l, 0)
            ui = min(i + self.u + 1, n)
            sparsity[i, li:ui] = 1

        super(BroydenBanded, self).__init__(n, n, specs, sparsity=sparsity)

    def fun(self, x):
        f = np.empty(self.n)
        for i in range(self.n):
            li = max(i - self.l, 0)
            ui = min(i + self.u + 1, self.n)
            a = x[li:i]
            b = x[i + 1:ui]
            s = np.sum(a * (1 + a)) + np.sum(b * (1 + b))
            f[i] = x[i] * (2 + 5 * x[i]**2) + 1 - s
        return f

    def jac(self, x):
        J = lil_matrix((self.n, self.n))
        for i in range(self.n):
            J[i, i] = 2 + 15 * x[i]**2
            li = max(i - self.l, 0)
            ui = min(i + self.u + 1, self.n)
            a = x[li:i]
            b = x[i + 1:ui]
            J[i, li:i] = -2 * a - 1
            J[i, i + 1:ui] = -2 * b - 1
        return J


class BroydenBanded10K(BroydenBanded):
    def __init__(self):
        super(BroydenBanded10K, self).__init__(10000, 2, 2)


class PowellBadlyScaled(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([0.0, 1.0])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0.0, 1.0], [1.0, 9.0]))
        ]
        super(PowellBadlyScaled, self).__init__(2, 2, specs)

    def fun(self, x):
        return np.array([1e4 * x[0] * x[1] - 1,
                         np.exp(-x[0]) + np.exp(-x[1]) - 1.0001])

    def jac(self, x):
        return np.array([
            [1e4 * x[1], 1e4 * x[0]],
            [-np.exp(-x[0]), -np.exp(-x[1])]
        ])


class BrownBadlyScaled(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([1.0, 1.0])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0, 3e-5], [1e6, 100]))
        ]
        super(BrownBadlyScaled, self).__init__(2, 3, specs)

    def fun(self, x):
        return np.array([x[0] - 1e6, x[1] - 2e-6, x[0] * x[1] - 2])

    def jac(self, x):
        return np.array([
            [1, 0],
            [0, 1],
            [x[1], x[0]]
        ])


class Beale(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([1.0, 1.0])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0.6, 0.5], [10.0, 100.0]))
        ]
        super(Beale, self).__init__(2, 3, specs)

    def fun(self, x):
        return np.array([1.5 - x[0] * (1 - x[1]),
                         2.25 - x[0] * (1 - x[1]**2),
                         2.625 - x[0] * (1 - x[1]**3)])

    def jac(self, x):
        return np.array([
            [-(1 - x[1]), x[0]],
            [-(1 - x[1]**2), 2 * x[0] * x[1]],
            [-(1 - x[1]**3), 3 * x[0] * x[1]**2]
        ])


class HelicalValley(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([-1.0, 0.0, 0.0])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([-100.0, -1, -1], [0.8, 1, 1]))
        ]
        super(HelicalValley, self).__init__(3, 3, specs)

    def fun(self, x):
        f = np.array([
            10 * (x[2] - 5 / np.pi * np.arctan(x[1] / x[0])),
            10 * ((x[0]**2 + x[1]**2)**0.5 - 1),
            x[2]
        ])
        if x[0] <= 0:
            f[0] -= 50
        return f

    def jac(self, x):
        d1 = 1 + (x[1] / x[0])**2
        d2 = (x[0]**2 + x[1]**2) ** 0.5
        return np.array([
            [50 / np.pi * x[1] / x[0]**2 / d1, -50 / np.pi / x[0] / d1, 10],
            [10 * x[0] / d2, 10 * x[1] / d2, 0],
            [0, 0, 1]
        ])


class GaussianFittingII(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([0.4, 1.0, 0.0])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0.398, 1, -0.5], [4.2, 2, 0.1]))
        ]
        super(GaussianFittingII, self).__init__(3, 15, specs)

        self.t = 0.5 * (7 - np.arange(self.m))
        self.y = np.array([9, 44, 175, 540, 1295, 2420, 3521, 3989,
                           3521, 2420, 1295, 540, 175, 44, 9]) * 1e-4

    def fun(self, x):
        return x[0] * np.exp(-0.5 * x[1] * (self.t - x[2]) ** 2) - self.y

    def jac(self, x):
        e = np.exp(-0.5 * x[1] * (self.t - x[2]) ** 2)
        J = np.empty((self.m, self.n))
        J[:, 0] = e
        J[:, 1] = -0.5 * x[0] * (self.t - x[2]) ** 2 * e
        J[:, 2] = x[0] * x[1] * (self.t - x[2]) * e
        return J


class GulfRnD(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([5, 2.5, 0.15])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, (np.zeros(3), 10 * np.ones(3)))
        ]
        super(GulfRnD, self).__init__(3, 100, specs)

        self.t = (1 + np.arange(self.m)) / 100.0
        self.y = 25 + (-50 * np.log(self.t)) ** (2 / 3)

    def fun(self, x):
        return np.exp(-np.abs(x[1] - self.y) ** x[2] / x[0]) - self.t

    def jac(self, x):
        J = np.empty((self.m, self.n))
        d = x[1] - self.y
        e = np.exp(-np.abs(d) ** x[2] / x[0])
        J[:, 0] = np.abs(d) ** x[2] * x[0]**-2 * e
        J[d >= 0, 1] = -x[2] * d[d >= 0] ** (x[2] - 1) / x[0]
        J[d < 0, 1] = x[2] * (-d[d < 0]) ** (x[2] - 1) / x[0]
        J[:, 1] *= e
        with np.errstate(invalid='ignore'):
            J[:, 2] = np.nan_to_num(
                -np.log(np.abs(d)) * np.abs(d) ** x[2] / x[0] * e)
        return J


class Box3D(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0_1 = np.array([0.0, 10, 20])
        x0_2 = np.array([0.0, 7.5, 20])
        specs = [
            (x0_1, (-np.inf, np.inf)),
            (x0_2, ([0.0, 5, 0], [2, 9.5, 20]))
        ]
        super(Box3D, self).__init__(3, 10, specs)
        self.t = 0.1 * (1 + np.arange(self.m))

    def fun(self, x):
        return (np.exp(-x[0] * self.t) - np.exp(-x[1] * self.t) -
                x[2] * (np.exp(-self.t) - np.exp(-10 * self.t)))

    def jac(self, x):
        J = np.empty((self.m, self.n))
        J[:, 0] = -self.t * np.exp(-x[0] * self.t)
        J[:, 1] = self.t * np.exp(-x[1] * self.t)
        J[:, 2] = -np.exp(-self.t) + np.exp(-10 * self.t)
        return J


class Wood(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([-3.0, -1, -3, -1])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, (-100 * np.ones(4), [0, 10, 100, 100]))
        ]
        super(Wood, self).__init__(4, 6, specs)

    def fun(self, x):
        return np.array([
            10 * (x[1] - x[0]**2),
            1 - x[0],
            90**0.5 * (x[3] - x[2]**2),
            1 - x[2],
            10**0.5 * (x[1] + x[3] - 2),
            10**-0.5 * (x[1] - x[3])
        ])

    def jac(self, x):
        return np.array([
            [-20 * x[0], 10, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -2 * 90**0.5 * x[2], 90**0.5],
            [0, 0, -1, 0],
            [0, 10**0.5, 0, 10**0.5],
            [0, 10**-0.5, 0, -10**-0.5]
        ])


class Biggs(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([1.0, 2, 1, 1, 1, 1])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0.0, 0, 0, 1, 0, 0], [2.0, 8, 1, 7, 5, 5]))
        ]
        super(Biggs, self).__init__(6, 13, specs)

        self.t = (1 + np.arange(self.m)) * 0.1
        self.y = (np.exp(-self.t) - 5 * np.exp(-10 * self.t) +
                  3 * np.exp(-4 * self.t))

    def fun(self, x):
        return (x[2] * np.exp(-x[0] * self.t) - x[3] * np.exp(-x[1] * self.t) +
                x[5] * np.exp(-x[4] * self.t) - self.y)

    def jac(self, x):
        J = np.empty((self.m, self.n))
        J[:, 0] = -x[2] * self.t * np.exp(-x[0] * self.t)
        J[:, 1] = x[3] * self.t * np.exp(-x[1] * self.t)
        J[:, 2] = np.exp(-x[0] * self.t)
        J[:, 3] = -np.exp(-x[1] * self.t)
        J[:, 4] = -x[5] * self.t * np.exp(-x[4] * self.t)
        J[:, 5] = np.exp(-x[4] * self.t)
        return J


class VariablyDimensional(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.x0 = 1 - (1 + np.arange(n)) / n
        specs = [
            (self.x0, (-np.inf, np.inf))
        ]
        super(VariablyDimensional, self).__init__(n, n + 2, specs)

    def fun(self, x):
        f = np.empty(self.m)
        f[:self.n] = x - 1
        v = np.sum((1 + np.arange(self.n)) * f[:self.n])
        f[-2] = v
        f[-1] = v**2
        return f

    def jac(self, x):
        J = np.empty((self.m, self.n))
        J[:self.n, :self.n] = np.identity(self.n)
        v = np.sum((1 + np.arange(self.n)) * (x - 1))
        J[-2] = 1 + np.arange(self.n)
        J[-1] = 2 * v * J[-2]
        return J


class VariablyDimensional10(VariablyDimensional):
    def __init_(self):
        super(VariablyDimensional10, self).__init__(10)
        self.specs += [
            (self.x0, (np.zeros(self.n),
                       [10, 20, 30, 40, 50, 60, 70, 80, 90, 0.5]))
        ]


class Watson(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.x0 = np.zeros(n)
        specs = []
        super(Watson, self).__init__(n, 31, specs)
        self.t = (1 + np.arange(29)) / 29

    def fun(self, x):
        t = self.t[:, np.newaxis]
        j = np.arange(1, self.n)
        s1 = np.sum(j * x[j] * t ** (j - 1), axis=1)

        j = np.arange(self.n)
        s2 = np.sum(x[j] * t ** j, axis=1)

        return np.hstack((s1 - s2**2 - 1, x[0], x[1] - x[0]**2 - 1))

    def jac(self, x):
        j = np.arange(self.n)
        s2 = np.sum(x[j] * self.t[:, np.newaxis] ** j, axis=1)

        J = np.zeros((self.m, self.n))
        J[:-2, 0] = -2 * s2

        j = np.arange(1, self.n)
        t = self.t[:, np.newaxis]
        J[:-2, 1:] = j * t ** (j - 1) - 2 * s2[:, np.newaxis] * t ** j

        J[-2, 0] = 1
        J[-1, 0] = - 2 * x[0]
        J[-1, 1] = 1
        return J


class Watson6(Watson):
    def __init__(self):
        super(Watson6, self).__init__(6)
        self.specs += [
            (self.x0, (-np.inf, np.inf))
        ]


class Watson9(Watson):
    def __init__(self):
        super(Watson9, self).__init__(9)
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (self.x0, ([-1e-5, 0, 0, 0, 0, -3, 0, -3, 0],
                       [1e-5, 0.9, 0.1, 1, 1, 0, 4, 0, 2]))
        ]


class Watson12(Watson):
    def __init__(self):
        super(Watson12, self).__init__(12)
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (self.x0, ([-1.0, 0, -1, -1,  -1, 0, -3,  0, -10,  0, -5, 0],
                       [0,   0.9, 0,  0.3, 0, 1,  0, 10,  0,  10,  0, 1]))
        ]


class Watson20(Watson):
    def __init__(self):
        super(Watson20, self).__init__(20)
        self.specs += [
            (self.x0, (-np.inf, np.inf))
        ]


class PenaltyI(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.x0 = np.arange(n) + 1.0
        specs = [
            (self.x0, (-np.inf, np.inf))
        ]
        super(PenaltyI, self).__init__(n, n + 1, specs, sparsity=None)

    def fun(self, x):
        return np.hstack((
            1e-5**0.5 * (x - 1), np.dot(x, x) - 0.25
        ))

    def jac(self, x):
        J = np.zeros((self.m, self.n))
        i = np.arange(self.n)
        J[i, i] = 1e-5**0.5
        J[-1, :] = 2 * x
        return J


class PenaltyI10(PenaltyI):
    def __init__(self):
        super(PenaltyI10, self).__init__(10)
        self.specs += [
            (self.x0, ([0.0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       np.ones(self.n) * 100))
        ]


class PenaltyII(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.x0 = 0.5 * np.ones(n)
        specs = []
        super(PenaltyII, self).__init__(n, 2 * n, specs)

        t = np.arange(1, self.n) * 0.1
        self.y = np.exp(t + 0.1) + np.exp(t)

    def fun(self, x):
        return np.hstack((
            x[0],
            1e-5**0.5 * (np.exp(0.1 * x[1:]) + np.exp(0.1 * x[:-1]) - self.y),
            1e-5**0.5 * (np.exp(0.1 * x[1:]) - np.exp(-0.1)),
            np.sum((1.0 + np.arange(self.n)) * x[::-1]**2) - 1
        ))

    def jac(self, x):
        J = np.zeros((self.m, self.n))
        J[0, 0] = 1.0
        i = np.arange(self.n - 1) + 1
        J[i, i - 1] += 1e-5**0.5 * 0.1 * np.exp(0.1 * x[i - 1])
        J[i, i] += 1e-5**0.5 * 0.1 * np.exp(0.1 * x[i])
        J[self.n - 1 + i, i] = 1e-5**0.5 * 0.1 * np.exp(0.1 * x[i])
        i = 1.0 + np.arange(self.n)
        J[-1, :] = 2 * i[::-1] * x
        return J


class PenaltyII4(PenaltyII):
    def __init__(self):
        super(PenaltyII4, self).__init__(4)
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (self.x0, ([-10, 0.3, 0, -1],
                       [50, 50, 50, 0.5]))
        ]


class PenaltyII10(PenaltyII):
    def __init__(self):
        super(PenaltyII10, self).__init__(10)
        self.specs += [
            (self.x0, (-np.inf, np.inf)),
            (self.x0, ([-10, 0.1, 0, 0.05, 0, -10, 0, 0.2, 0, 0],
                       np.hstack((50 * np.ones(9), 0.5))))
        ]


class BrownAndDennis(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([25.0, 5, -5, -1])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([-10.0, 0, -100, -20], [100, 15, 0, 0.2]))
        ]
        super(BrownAndDennis, self).__init__(4, 20, specs)

        self.t = (1.0 + np.arange(self.m)) / 5

    def fun(self, x):
        return ((x[0] + x[1] * self.t - np.exp(self.t)) ** 2 +
                (x[2] + x[3] * np.sin(self.t) - np.cos(self.t)) ** 2)

    def jac(self, x):
        J = np.empty((self.m, self.n))
        f1 = x[0] + x[1] * self.t - np.exp(self.t)
        f2 = x[2] + x[3] * np.sin(self.t) - np.cos(self.t)
        J[:, 0] = 2 * f1
        J[:, 1] = 2 * self.t * f1
        J[:, 2] = 2 * f2
        J[:, 3] = 2 * np.sin(self.t) * f2
        return J


class Trigonometric(LSQBenchmarkProblemFactory):
    def __init__(self):
        n = 10
        x0 = np.ones(n) / n
        x1 = 5.0 + 10 * np.arange(10)
        specs = [
            (x0, (-np.inf, np.inf)),
            (x1, (10 * np.arange(10), 10 * np.arange(10) + 10))
        ]
        super(Trigonometric, self).__init__(n, n, specs)

    def fun(self, x):
        cosx = np.cos(x)
        i = np.arange(self.n) + 1.0
        return self.n + i * (1 - cosx) - np.sin(x) - np.sum(cosx)

    def jac(self, x):
        J = np.zeros((self.m, self.n))
        i = np.arange(self.n)
        J[i, i] = (i + 1) * np.sin(x) - np.cos(x)
        J += np.sin(x)
        return J


# class TrigonometricBadRank(Trigonometric):
#     def __init__(self):
#         super(TrigonometricBadRank, self).__init__()
#
#     def fun(self, x):
#         f = super(TrigonometricBadRank, self).fun(x)
#         return f[:-1]
#
#     def jac(self, x):
#         J = super(TrigonometricBadRank, self).jac(x)
#         return J[:-1]


class PowellSingular(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([3.0, -1, 0, 1])
        specs = [
            (x0, (-np.inf, np.inf)),
            (x0, ([0.1, -20, -1, -1], [100.0, 20, 1, 50]))
        ]
        super(PowellSingular, self).__init__(4, 4, specs)

    def fun(self, x):
        return np.array([
            x[0] + 10 * x[1], 5**0.5 * (x[2] - x[3]),
            (x[1] - 2 * x[2]) ** 2, 10**0.5 * (x[0] - x[3]) ** 2
        ])

    def jac(self, x):
        return np.array([
            [1, 10, 0, 0],
            [0, 0, 5**0.5, -5**0.5],
            [0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0],
            [2 * 10**0.5 * (x[0] - x[3]), 0, 0, -2 * 10**0.5 * (x[0] - x[3])]
        ])


class PowellSingularExtended(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        if n % 4 != 0:
            raise ValueError("`n` must be divisible by 4.")
        self.x0 = np.empty(n)
        self.x0[::4] = 3
        self.x0[1::4] = -1
        self.x0[2::4] = 0
        self.x0[3::4] = 1
        specs = [
            (self.x0, (-np.inf, np.inf)),
        ]

        sparsity = lil_matrix((n, n), dtype=int)
        i0 = np.arange(0, n, 4)
        i1 = np.arange(1, n, 4)
        i2 = np.arange(2, n, 4)
        i3 = np.arange(3, n, 4)
        sparsity[i0, i0] = 1
        sparsity[i0, i1] = 1
        sparsity[i1, i2] = 1
        sparsity[i1, i3] = 1
        sparsity[i2, i1] = 1
        sparsity[i2, i2] = 1
        sparsity[i3, i0] = 1
        sparsity[i3, i3] = 1

        super(PowellSingularExtended, self).__init__(n, n, specs,
                                                     sparsity=sparsity)

    def fun(self, x):
        f = np.empty(self.n)
        f[::4] = x[::4] + 10 * x[1::4]
        f[1::4] = 5**0.5 * (x[2::4] - x[3::4])
        f[2::4] = (x[1::4] - 2 * x[2::4])**2
        f[3::4] = 10**0.5 * (x[::4] - x[3::4])**2
        return f

    def jac(self, x):
        J = lil_matrix((self.n, self.n))
        i0 = np.arange(0, self.n, 4)
        i1 = np.arange(1, self.n, 4)
        i2 = np.arange(2, self.n, 4)
        i3 = np.arange(3, self.n, 4)
        J[i0, i0] = 1
        J[i0, i1] = 10
        J[i1, i2] = 5**0.5
        J[i1, i3] = -5**0.5
        J[i2, i1] = 2 * (x[i1] - 2 * x[i2])
        J[i2, i2] = -4 * (x[i1] - 2 * x[i2])
        J[i3, i0] = 2 * 10**0.5 * (x[i0] - x[i3])
        J[i3, i3] = -2 * 10**0.5 * (x[i0] - x[i3])
        return J


class PowellSingularExtended10K(PowellSingularExtended):
    def __init__(self):
        super(PowellSingularExtended10K, self).__init__(10000)


class FreudensteinAndRoth(LSQBenchmarkProblemFactory):
    def __init__(self):
        x0 = np.array([-0.5, 2])
        specs = [
            (x0, (-np.inf, np.inf))
        ]
        super(FreudensteinAndRoth, self).__init__(2, 2, specs)

    def fun(self, x):
        return np.array([
            -13.0 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1],
            -29 + x[0] + ((x[1] + 1) * x[1] - 14) * x[1]
        ])

    def jac(self, x):
        return np.array([
            [1.0, 10 * x[1] - 3*x[1]**2 - 2],
            [1, 3 * x[1]**2 + 2*x[1] - 14]
        ])


class JenrichAndSampson(LSQBenchmarkProblemFactory):
    def __init__(self, m):
        self.x0 = np.array([0.3, 0.4])
        specs = []
        super(JenrichAndSampson, self).__init__(2, m, specs)

    def fun(self, x):
        i = np.arange(self.m) + 1.0
        return np.exp(x[0] * i) + np.exp(x[1] * i) - 2 * (i + 1)

    def jac(self, x):
        i = np.arange(self.m) + 1.0
        J = np.empty((self.m, self.n))
        J[:, 0] = i * np.exp(x[0] * i)
        J[:, 1] = i * np.exp(x[1] * i)
        return J


class JenrichAndSampson10(JenrichAndSampson):
    def __init__(self):
        super(JenrichAndSampson10, self).__init__(10)
        self.specs += [
            (self.x0, (-np.inf, np.inf))
        ]


class DiscreteBoundaryValue(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.h = 1 / (n + 1)
        self.t = (1 + np.arange(n)) * self.h
        self.x0 = self.t * (self.t - 1)
        specs = [
            (self.x0, (-np.inf, np.inf))
        ]

        sparsity = lil_matrix((n, n), dtype=int)
        i = np.arange(n)
        sparsity[i, i] = 1
        i = np.arange(n - 1)
        sparsity[i, i + 1] = 1
        i = np.arange(1, n)
        sparsity[i, i - 1] = 1

        super(DiscreteBoundaryValue, self).__init__(n, n, specs,
                                                           sparsity=sparsity)

    def fun(self, x):
        f = 2 * x + 0.5 * self.h**2 * (x + self.t + 1)**3
        f[:-1] -= x[1:]
        f[1:] -= x[:-1]
        return f

    def jac(self, x):
        J = lil_matrix((self.n, self.n))
        i = np.arange(self.n)
        J[i, i] = 2 + 1.5 * self.h**2 * (x + self.t + 1)**2
        i = np.arange(self.n - 1)
        J[i, i + 1] = -1
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        return J


class DiscreteBoundaryValue10K(DiscreteBoundaryValue):
    def __init__(self):
        super(DiscreteBoundaryValue10K, self).__init__(10000)


class BroydenTridiagonal(LSQBenchmarkProblemFactory):
    def __init__(self, n):
        self.x0 = -np.ones(n)
        specs = [
            (self.x0, (-np.inf, np.inf))
        ]

        sparsity = lil_matrix((n, n), dtype=int)
        i = np.arange(n)
        sparsity[i, i] = 1
        i = np.arange(1, n)
        sparsity[i, i - 1] = 1
        i = np.arange(n - 1)
        sparsity[i, i + 1] = 1

        super(BroydenTridiagonal, self).__init__(n, n, specs,
                                                 sparsity=sparsity)

    def fun(self, x):
        f = (3 - x) * x + 1
        f[1:] -= x[:-1]
        f[:-1] -= 2 * x[1:]
        return f

    def jac(self, x):
        J = lil_matrix((self.n, self.n))
        i = np.arange(self.n)
        J[i, i] = 3 - 2 * x
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        i = np.arange(self.n - 1)
        J[i, i + 1] = -2
        return J


class BroydenTridiagonal10K(BroydenTridiagonal):
    def __init__(self):
        super(BroydenTridiagonal10K, self).__init__(10000)


# NIST nonlinear regression problems.


thisdir, thisfile = os.path.split(__file__)
NIST_DIR = os.path.join(thisdir, 'NIST_STRD')


def read_nist_data(dataset):
    """NIST STRD data is in a simple, fixed format with
    line numbers being significant!
    """
    finp = open(os.path.join(NIST_DIR, "%s.dat" % dataset), 'r')
    lines = [l[:-1] for l in finp.readlines()]
    finp.close()
    ModelLines = lines[30:39]
    ParamLines = lines[40:58]
    DataLines = lines[60:]

    words = ModelLines[1].strip().split()
    nparams = int(words[0])

    start1 = [0]*nparams
    start2 = [0]*nparams
    certval = [0]*nparams
    certerr = [0]*nparams
    for i, text in enumerate(ParamLines[:nparams]):
        [s1, s2, val, err] = [float(x) for x in text.split('=')[1].split()]
        start1[i] = s1
        start2[i] = s2
        certval[i] = val
        certerr[i] = err

    #
    for t in ParamLines[nparams:]:
        t = t.strip()
        if ':' not in t:
            continue
        val = float(t.split(':')[1])
        if t.startswith('Residual Sum of Squares'):
            sum_squares = val
        elif t.startswith('Residual Standard Deviation'):
            std_dev = val
        elif t.startswith('Degrees of Freedom'):
            nfree = int(val)
        elif t.startswith('Number of Observations'):
            ndata = int(val)

    y, x = [], []
    for d in DataLines:
        vals = [float(i) for i in d.strip().split()]
        y.append(vals[0])
        if len(vals) > 2:
            x.append(vals[1:])
        else:
            x.append(vals[1])

    y = np.array(y)
    x = np.array(x)
    out = {'y': y, 'x': x, 'nparams': nparams, 'ndata': ndata,
           'nfree': nfree, 'start1': start1, 'start2': start2,
           'sum_squares': sum_squares, 'std_dev': std_dev,
           'cert': certval,  'cert_values': certval,  'cert_stderr': certerr}
    return out


class NISTProblemFactory(LSQBenchmarkProblemFactory):
    def __init__(self, problem_name):
        data = read_nist_data(problem_name)
        m = data['ndata']
        n = data['nparams']
        self.x = data['x']
        self.y = data['y']

        specs = [
            (data['start1'], (-np.inf, np.inf)),
            (data['start2'], (-np.inf, np.inf))
        ]

        super(NISTProblemFactory, self).__init__(n, m, specs)


class NIST_Bennett5(NISTProblemFactory):
    def __init__(self):
        super(NIST_Bennett5, self).__init__('Bennett5')

    def fun(self, b):
        return b[0] * (b[1] + self.x)**(-1/b[2]) - self.y

    def jac(self, b):
        t = b[1] + self.x
        return np.vstack((
            (b[1] + self.x)**(-1/b[2]),
            -b[0] / b[2] * (b[1] + self.x)**(-1/b[2] - 1),
            b[0] / b[2]**2 * (b[1] + self.x)**(-1/b[2]) * np.log(t)
        )).T


class NIST_BoxBOD(NISTProblemFactory):
    def __init__(self):
        super(NIST_BoxBOD, self).__init__('BoxBOD')

    def fun(self, b):
        return b[0] * (1 - np.exp(-b[1] * self.x)) - self.y

    def jac(self, b):
        return np.vstack((
            1 - np.exp(-b[1] * self.x),
            b[0] * self.x * np.exp(-b[1] * self.x)
        )).T


class NIST_Chwirut(NISTProblemFactory):
    def __init__(self, i):
        super(NIST_Chwirut, self).__init__('Chwirut' + str(i))

    def fun(self, b):
        return np.exp(-b[0] * self.x) / (b[1] + b[2] * self.x) - self.y

    def jac(self, b):
        denom = b[1] + b[2] * self.x
        return np.vstack((
            -self.x * np.exp(-b[0] * self.x) / denom,
            -np.exp(-b[0] * self.x) / denom**2,
            -self.x * np.exp(-b[0] * self.x) / denom**2
        )).T


class NIST_Chwirut1(NIST_Chwirut):
    def __init__(self):
        super(NIST_Chwirut1, self).__init__(1)


class NIST_Chwirut2(NIST_Chwirut):
    def __init__(self):
        super(NIST_Chwirut2, self).__init__(2)


class NIST_DanWood(NISTProblemFactory):
    def __init__(self):
        super(NIST_DanWood, self).__init__('DanWood')

    def fun(self, b):
        return b[0] * self.x**b[1] - self.y

    def jac(self, b):
        return np.vstack((
            self.x**b[1],
            b[0] * np.log(self.x) * self.x**b[1]
        )).T


class NIST_ENSO(NISTProblemFactory):
    def __init__(self):
        super(NIST_ENSO, self).__init__('ENSO')

    def fun(self, b):
        return (b[0] + b[1] * np.cos(2 * np.pi * self.x / 12) +
                b[2] * np.sin(2 * np.pi * self.x / 12) +
                b[4] * np.cos(2 * np.pi * self.x / b[3]) +
                b[5] * np.sin(2 * np.pi * self.x / b[3]) +
                b[7] * np.cos(2 * np.pi * self.x / b[6]) +
                b[8] * np.sin(2 * np.pi * self.x / b[6]) - self.y)

    def jac(self, b):
        return np.vstack((
            np.ones_like(self.x),
            np.cos(2 * np.pi * self.x / 12),
            np.sin(2 * np.pi * self.x / 12),
            2 * np.pi * self.x / b[3]**2 * (
                b[4] * np.sin(2 * np.pi * self.x / b[3]) -
                b[5] * np.cos(2 * np.pi * self.x / b[3])),
            np.cos(2 * np.pi * self.x / b[3]),
            np.sin(2 * np.pi * self.x / b[3]),
            2 * np.pi * self.x / b[6]**2 * (
                b[7] * np.sin(2 * np.pi * self.x / b[6]) -
                b[8] * np.cos(2 * np.pi * self.x / b[6])),
            np.cos(2 * np.pi * self.x / b[6]),
            np.sin(2 * np.pi * self.x / b[6])
        )).T


class NIST_Eckerle4(NISTProblemFactory):
    def __init__(self):
        super(NIST_Eckerle4, self).__init__('Eckerle4')

    def fun(self, b):
        return b[0] / b[1] * np.exp(-0.5*((self.x - b[2]) / b[1])**2) - self.y

    def jac(self, b):
        e = np.exp(-0.5*((self.x - b[2]) / b[1])**2)
        return np.vstack((
            e / b[1],
            b[0] * e / b[1]**2 * (((self.x - b[2]) / b[1])**2 - 1),
            b[0] * (self.x - b[2]) * e / b[1]**3
        )).T


class NIST_Gauss(NISTProblemFactory):
    def __init__(self, i):
        super(NIST_Gauss, self).__init__("Gauss" + str(i))

    def fun(self, b):
        return (b[0] * np.exp(-b[1] * self.x) +
                b[2] * np.exp(-(self.x - b[3])**2 / b[4]**2) +
                b[5] * np.exp(-(self.x - b[6])**2 / b[7]**2) - self.y)

    def jac(self, b):
        return np.vstack((
            np.exp(-b[1] * self.x),
            -b[0] * self.x * np.exp(-b[1] * self.x),
            np.exp(-(self.x - b[3])**2 / b[4]**2),
            2 * b[2] * (self.x - b[3]) / b[4]**2 *
            np.exp(-(self.x - b[3])**2 / b[4]**2),
            2 * b[2] * (self.x - b[3])**2 / b[4]**3 *
            np.exp(-(self.x - b[3])**2 / b[4]**2),
            np.exp(-(self.x - b[6])**2 / b[7]**2),
            2 * b[5] * (self.x - b[6]) / b[7]**2 *
            np.exp(-(self.x - b[6])**2 / b[7]**2),
            2 * b[5] * (self.x - b[6])**2 / b[7]**3 *
            np.exp(-(self.x - b[6])**2 / b[7]**2)
        )).T


class NIST_Gauss1(NIST_Gauss):
    def __init__(self):
        super(NIST_Gauss1, self).__init__(1)


class NIST_Gauss2(NIST_Gauss):
    def __init__(self):
        super(NIST_Gauss2, self).__init__(2)


class NIST_Gauss3(NIST_Gauss):
    def __init__(self):
        super(NIST_Gauss3, self).__init__(3)


class NIST_Hahn1(NISTProblemFactory):
    def __init__(self):
        super(NIST_Hahn1, self).__init__("Hahn1")

    def fun(self, b):
        return (
            (b[0] + b[1] * self.x + b[2] * self.x**2 + b[3] * self.x**3) /
            (1 + b[4] * self.x + b[5] * self.x**2 + b[6] * self.x**3) -
            self.y)

    def jac(self, b):
        numer = b[0] + b[1] * self.x + b[2] * self.x**2 + b[3] * self.x**3
        denom = 1 + b[4] * self.x + b[5] * self.x**2 + b[6] * self.x**3

        return np.vstack((
            1 / denom,
            self.x / denom,
            self.x**2 / denom,
            self.x**3 / denom,
            -self.x * numer / denom**2,
            -self.x**2 * numer / denom**2,
            -self.x**3 * numer / denom**2
        )).T


class NIST_Kirby2(NISTProblemFactory):
    def __init__(self):
        super(NIST_Kirby2, self).__init__('Kirby2')

    def fun(self, b):
        return ((b[0] + b[1] * self.x + b[2] * self.x**2) /
                (1 + b[3] * self.x + b[4] * self.x**2) - self.y)

    def jac(self, b):
        numer = b[0] + b[1] * self.x + b[2] * self.x**2
        denom = 1 + b[3] * self.x + b[4] * self.x**2
        return np.vstack((
            1 / denom,
            self.x / denom,
            self.x**2 / denom,
            -self.x * numer / denom**2,
            -self.x**2 * numer / denom**2
        )).T


class NIST_Lanczos(NISTProblemFactory):
    def __init__(self, i):
        super(NIST_Lanczos, self).__init__('Lanczos' + str(i))

    def fun(self, b):
        return (b[0] * np.exp(-b[1] * self.x) +
                b[2] * np.exp(-b[3] * self.x) +
                b[4] * np.exp(-b[5] * self.x) - self.y)

    def jac(self, b):
        return np.vstack((
            np.exp(-b[1] * self.x),
            -b[0] * self.x * np.exp(-b[1] * self.x),
            np.exp(-b[3] * self.x),
            -b[2] * self.x * np.exp(-b[3] * self.x),
            np.exp(-b[5] * self.x),
            -b[4] * self.x * np.exp(-b[5] * self.x)
        )).T


class NIST_Lanczos1(NIST_Lanczos):
    def __init__(self):
        super(NIST_Lanczos1, self).__init__(1)


class NIST_Lancsoz2(NIST_Lanczos):
    def __init__(self):
        super(NIST_Lancsoz2, self).__init__(2)


class NIST_Lancsoz3(NIST_Lanczos):
    def __init__(self):
        super(NIST_Lancsoz3, self).__init__(3)


class NIST_MGH09(NISTProblemFactory):
    def __init__(self):
        super(NIST_MGH09, self).__init__('MGH09')

    def fun(self, b):
        return (b[0] * (self.x**2 + b[1] * self.x[1]) /
                (self.x**2 + b[2] * self.x + b[3]) - self.y)

    def jac(self, b):
        numer = self.x**2 + b[1] * self.x[1]
        denom = self.x**2 + b[2] * self.x + b[3]
        return np.vstack((
            numer / denom,
            b[0] * self.x[1] / denom,
            -b[0] * self.x * numer / denom**2,
            -b[0] * numer / denom**2
        )).T


class NIST_MGH10(NISTProblemFactory):
    def __init__(self):
        super(NIST_MGH10, self).__init__('MGH10')

    def fun(self, b):
        return b[0] * np.exp(b[1] / (self.x + b[2])) - self.y

    def jac(self, b):
        e = np.exp(b[1] / (self.x + b[2]))
        return np.vstack((
            e,
            b[0] / (self.x + b[2]) * e,
            -b[0] * b[1] / (self.x + b[2])**2 * e
        )).T


class NIST_MGH17(NISTProblemFactory):
    def __init__(self):
        super(NIST_MGH17, self).__init__('MGH17')

    def fun(self, b):
        return (b[0] + b[1] * np.exp(-b[3] * self.x) +
                b[2] * np.exp(-b[4] * self.x) - self.y)

    def jac(self, b):
        return np.vstack((
            np.ones_like(self.x),
            np.exp(-b[3] * self.x),
            np.exp(-b[4] * self.x),
            -b[1] * self.x * np.exp(-b[3] * self.x),
            -b[2] * self.x * np.exp(-b[4] * self.x)
        )).T


class NIST_Misra1a(NISTProblemFactory):
    def __init__(self):
        super(NIST_Misra1a, self).__init__("Misra1a")

    def fun(self, b):
        return b[0] * (1 - np.exp(-b[1] * self.x)) - self.y

    def jac(self, b):
        return np.vstack((
            1 - np.exp(-b[1] * self.x),
            b[0] * self.x * np.exp(-b[1] * self.x)
        )).T


class NIST_Misra1b(NISTProblemFactory):
    def __init__(self):
        super(NIST_Misra1b, self).__init__("Misra1b")

    def fun(self, b):
        return b[0] * (1 - (1 + 0.5 * b[1] * self.x)**-2) - self.y

    def jac(self, b):
        return np.vstack((
            1 - (1 + 0.5 * b[1] * self.x)**-2,
            b[0] * self.x * (1 + 0.5 * b[1] * self.x)**-3
        )).T


class NIST_Misra1c(NISTProblemFactory):
    def __init__(self):
        super(NIST_Misra1c, self).__init__("Misra1c")

    def fun(self, b):
        return b[0] * (1 - (1 + 2 * b[1] * self.x)**-0.5) - self.y

    def jac(self, b):
        return np.vstack((
            1 - (1 + 2 * b[1] * self.x)**-0.5,
            b[0] * self.x * (1 + 2 * b[1] * self.x)**-1.5
        )).T


class NIST_Misra1d(NISTProblemFactory):
    def __init__(self):
        super(NIST_Misra1d).__init__("Misra1d")

    def fun(self, b):
        return b[0] * b[1] * self.x / (1 + b[1] * self.x) - self.y

    def jac(self, b):
        return np.vstack((
            b[1] * self.x / (1 + b[1] * self.x),
            b[0] * self.x / (1 + b[1] * self.x) *
            (1 - b[1] * self.x / (1 + b[1] * self.x))
        )).T


class NIST_Nelson(NISTProblemFactory):
    def __init__(self):
        super(NIST_Nelson, self).__init__("Nelson")

    def fun(self, b):
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]

        return b[0] - b[1] * x1 * np.exp(-b[2] * x2) - self.y

    def jac(self, b):
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]

        return np.vstack((
            np.ones_like(x1),
            -x1 * np.exp(-b[2] * x2),
            b[1] * x1 * x2 * np.exp(-b[2] * x2)
        )).T


class NIST_Rat42(NISTProblemFactory):
    def __init__(self):
        super(NIST_Rat42, self).__init__("Rat42")

    def fun(self, b):
        return b[0] / (1 + np.exp(b[1] - b[2] * self.x)) - self.y

    def jac(self, b):
        e = np.exp(b[1] - b[2] * self.x)
        return np.vstack((
            1 / (1 + e),
            -b[0] * e / (1 + e)**2,
            b[0] * self.x * e / (1 + e)**2
        )).T


class NIST_Rat43(NISTProblemFactory):
    def __init__(self):
        super(NIST_Rat43, self).__init__("Rat43")

    def fun(self, b):
        return b[0] * (1 + np.exp(b[1] - b[2] * self.x))**(-1 / b[3]) - self.y

    def jac(self, b):
        e = np.exp(b[1] - b[2] * self.x)
        return np.vstack((
            (1 + e)**(-1 / b[3]),
            -b[0] / b[3] * e * (1 + e)**(-1 / b[3] - 1),
            b[0] / b[3] * self.x * e * (1 + e)**(-1 / b[3] - 1),
            b[0] / b[3]**2 * np.log(1 + e) * (1 + e)**(-1 / b[3])
        )).T


class NIST_Roszman1(NISTProblemFactory):
    def __init__(self):
        super(NIST_Roszman1).__init__("Roszman1")

    def fun(self, b):
        return (b[0] - b[1] * self.x -
                np.arctan(b[2] / (self.x - b[3])) / np.pi - self.y)

    def jac(self, b):
        t = b[2] / (self.x - b[3])
        return np.vstack((
            np.ones_like(self.x),
            -self.x,
            -1 / (np.pi * (1 + t**2) * (self.x - b[3])),
            -b[2] / (np.pi * (1 + t**2) * (self.x - b[3])**2)
        )).T


class NIST_Thurber(NISTProblemFactory):
    def __init__(self):
        super(NIST_Thurber).__init__("Thurber")

    def fun(self, b):
        return ((b[0] + b[1] * self.x + b[2] * self.x**2 + b[3] * self.x**3) /
                (1 + b[4] * self.x + b[5] * self.x**2 + b[6] * self.x**3) -
                self.y)

    def jac(self, b):
        numer = b[0] + b[1] * self.x + b[2] * self.x**2 + b[3] * self.x**3
        denom = 1 + b[4] * self.x + b[5] * self.x**2 + b[6] * self.x**3
        return np.vstack((
            1 / denom,
            self.x / denom,
            self.x**2 / denom,
            self.x**3 / denom,
            -self.x * numer / denom**2,
            -self.x**2 * numer / denom**2,
            -self.x**3 * numer / denom**2
        )).T


def extract_lsq_problems():
    unbounded = []
    bounded = []
    sparse = []
    for name, factory in inspect.getmembers(sys.modules[__name__],
                                            inspect.isclass):
        if (name != "LSQBenchmarkProblemFactory" and
                issubclass(factory, LSQBenchmarkProblemFactory)):
            try:
                f = factory()
                u, b = f.extract_problems()
                if f.sparsity is not None:
                    sparse += b
                    sparse += u
                else:
                    bounded += b
                    unbounded += u
            except TypeError:
                pass

    return unbounded, bounded, sparse


if __name__ == '__main__':
    u, b, s = extract_lsq_problems()
    for p in u:
        print(p.name, p.check_jacobian())
