"""
Stochastic reaction simulation
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
from numpy.random import multinomial


class Model(object):

    def __init__(
            self,
            variable_names: List[str],
            rate_constants: List[float],
            inits: List[float],
            transition_matrix: np.ndarray,
            propensity
    ):
        """

        :param variable_names: list of strings
        :param rate_constants: list of fixed rate parameters
        :param inits: list of initial values of variables
        :param transition_matrix:
        :param propensity: list of lambda functions of the form:
            lambda r,ini: some function of rates ans inits.
        """
        self.vn = variable_names
        self.rates = rate_constants
        self.inits = inits
        self.tm = transition_matrix
        self.pv = propensity#[compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl = len(self.pv) #length of propensity vector
        self.nvars = len(self.inits) #number of variables
        self.time = np.zeros(1)
        self.series = np.zeros(1)
        self.steps = 0
        self.res = None

    def run(
            self,
            method: str = 'SSA',
            tmax: int = 10,
            reps: int = 1
    ):
        res = np.zeros(
            (tmax, self.nvars, reps),
            dtype=float
        )
        tvec = np.arange(tmax)
        self.res = res

        if method =='SSA':
            for i in range(0, reps):
                steps = self.GSSA(tmax, i)

        self.time = tvec
        self.series = self.res
        self.steps = steps

    def getStats(
            self
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        return self.time, self.series, self.steps

    def GSSA(
            self,
            tmax: int = 50,
            round: int = 0
    ) -> int:
        """
        Gillespie Direct algorithm

        :param tmax:
        :param round:
        :return:
        """
        ini = self.inits
        r = self.rates
        pvi = self.pv
        l=self.pvl
        pv = np.zeros(
            l,
            dtype=float
        )
        tm = self.tm
        #tvec = np.arange(tmax,dtype=int)
        tc = 0
        steps = 0
        self.res[0, :, round] = ini
        a0 = 1.0
        for tim in range(1, tmax):
            while tc < tim:
                for i in range(l):
                    pv[i] = pvi[i](r, ini)
                a0 = np.sum(pv)
                tau = (-1/a0) * np.log(np.random.random())
                event = multinomial(1, (pv/a0) ) # event which will happen on this iteration
                ini += tm[:, event.nonzero()[0][0]]
                tc += tau
                steps += 1
                if a0 == 0:
                    break
            self.res[tim, :, round] = ini
            if a0 == 0:
                break
        return steps

    def CR(
            self,
            pv
    ):
        """
        Composition reaction algorithm
        """
        pass


def main():
    import numpy as np
    import time

    import mfm.math.reaction._reaction as reaction
    vars = ['s','i','r']
    ini= np.array([500, 1, 0], dtype=int)
    rates = np.array([.001, .1], dtype=float)
    tm = np.array(
        [
            [-1, 0],
            [1, -1],
            [0, 1]
        ]
    )

    prop = [
        reaction.l1,
        reaction.l2
    ]

    M = reaction.Model(
        vnames=vars,
        rates=rates,
        inits=ini,
        tmat=tm,
        propensity=prop
    )
    t0 = time.time()
    M.run(
        tmax=80,
        reps=1000
    )
    print('total time: ', (time.time()-t0))


