"""
Stochastic reaction simulation
"""
from __future__ import annotations
from chisurf import typing

import numpy as np
from numpy.random import multinomial


class Model(object):
    """
    Model class for simulating stochastic processes using algorithms such as SSA
    (Gillespie Direct) and related methods.

    This class is designed to facilitate numerical simulation of stochastic
    processes based on different reaction algorithms. It initializes a set of
    variables, rates, and configuration matrices, and provides methods for running
    simulations and analyzing results. The class can simulate reactions using the
    Stochastic Simulation Algorithm as well as other methods.

    Attributes
    ----------
    vn : List[str]
        The list of variable names in the model.
    rates : List[float]
        The rate constants (fixed parameter values).
    inits : List[float]
        Initial values of the variables in the system.
    tm : np.ndarray
        Transition matrix that defines state transitions.
    pv : List[Callable]
        List of propensity functions provided as lambda functions.
    pvl : int
        The length of the propensity vector.
    nvars : int
        Number of variables in the system.
    time : np.ndarray
        Array storing time points in the simulation.
    series : np.ndarray
        Array storing series output of the simulation.
    steps : int
        Number of steps executed in the simulation.
    res : Optional[np.ndarray]
        Final results of the simulation stored as a multi-dimensional array.
    """
    def __init__(
            self,
            variable_names: typing.List[str],
            rate_constants: typing.List[float],
            inits: typing.List[float],
            transition_matrix: np.ndarray,
            propensity
    ):
        """
        Summary:
        Initializes an object with specified dynamic system parameters such as the names of 
        variables, rate constants, initial conditions, transition matrix, and propensity 
        functions. The inputs are processed and stored as attributes of the object, while 
        derived properties like the number of variables, propensity vector length, and 
        tracked simulation values are also initialized.

        Attributes:
        variable_names : List[str]
            Names of the variables in the dynamic system.
        rate_constants : List[float]
            Numerical values for reaction rate constants corresponding to the system.
        inits : List[float]
            Initial conditions for the variables in the dynamic system.
        transition_matrix : np.ndarray
            Matrix defining state transitions within the dynamic system.
        propensity : List[str]
            List of propensity relationships or expressions evaluated within the system.

        Parameters:
        variable_names : List[str]
            Represents names of the variables in the system.
        rate_constants : List[float]
            Represents rates associated with transitions, reactions, or processes in 
            the dynamic system.
        inits : List[float]
            Represents the initial state values of the variables in the system.
        transition_matrix : np.ndarray
            Defines transitions between states in the variable space.
        propensity : List
            Contains definitions for propensity functions or expressions for the system.

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
        """
            Runs a simulation or modeling process based on the specified method and 
            parameters. This function facilitates the execution of the process using 
            the chosen algorithm and stores the results of the simulation in class 
            attributes for further analysis.

            Parameters
            ----------
            method : str, optional
                The method to be used for the simulation. Default is 'SSA'.
            tmax : int, optional
                The maximum number of time steps for the simulation. Default is 10.
            reps : int, optional
                The number of repetitions for the simulation. Default is 1.

            Attributes
            ----------
            res : numpy.ndarray
                A 3-dimensional array storing simulation results with dimensions 
                corresponding to time steps, variables, and repetitions.
            tvec : numpy.ndarray
                A 1-dimensional array representing the time steps.
            time : numpy.ndarray
                An array containing the time steps over which the simulation was 
                conducted.
            series : numpy.ndarray
                A 3-dimensional array, identical to 'res', representing simulation 
                results.
            steps : Any
                Results or metrics associated with the steps generated by the selected 
                method.
        """
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

    def getStats(self) -> typing.Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        """
            Retrieves statistical data components including time, series, and steps. This
            method consolidates and returns these components as a tuple to provide a
            structured representation of the related data.

            Returns
            -------
            tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
                A tuple containing time, series, and steps data in the form of
                numpy arrays.
        """
        return self.time, self.series, self.steps

    def GSSA(
            self,
            tmax: int = 50,
            round: int = 0
    ) -> int:
        """
        Simulates the Gillespie Stochastic Simulation Algorithm (GSSA) for a given number of timesteps and rounds.

        The GSSA is a method used to simulate the time evolution of a system of reactions stochastically.
        At each time step, the function calculates the propensity of each reaction, determines the time to
        the next reaction, and randomly selects the next reaction to occur. The simulation updates the state
        of the system and tracks its evolution over time. The simulation stops if no reactions can occur or
        the maximum number of timesteps is reached.

        Arguments:
            tmax (int): Number of timesteps to simulate. Defaults to 50.
            round (int): The current round of the simulation (index in a multidimensional result array).
                         Defaults to 0.

        Returns:
            int: The total number of reaction steps completed during the simulation.
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
        Composition reaction algorithm.
    
        The Composition Reaction (CR) algorithm simulates reactions efficiently by 
        segmenting the propensity vector into groups and computing reaction events 
        within these groups. It selects a group based on the sum of propensities and 
        simulates reaction events to determine when and where reactions occur.
    
        Parameters
        ----------
        pv : np.ndarray
            Propensity vector containing the computed probabilities of each reaction 
            occurring.
    
        Returns
        -------
        int
            Number of steps (reaction events) carried out during this simulation round.
        """
        ini = self.inits
        r = self.rates
        l = self.pvl
        tm = self.tm
        tc = 0
        steps = 0
        self.res[0, :, 0] = ini
        a_total = 1.0

        while a_total > 0:
            # Recompute the propensity vector
            for i in range(l):
                pv[i] = self.pv[i](r, ini)
            a_total = np.sum(pv)

            # If no reactions can occur, break out
            if a_total == 0:
                break

            # Determine the time increment tau
            tau = (-1 / a_total) * np.log(np.random.random())
            tc += tau

            # Select the next reaction to occur
            selected_reaction = np.random.choice(range(l), p=pv / a_total)
            ini += tm[:, selected_reaction]
            steps += 1

            # Record the system's state at the current time point
            self.res[min(int(tc), self.res.shape[0] - 1), :, 0] = ini

        return steps


def main():
    """
    Main function to simulate a reaction model and measure execution time.

    This function utilizes the `chisurf.math.reaction` module to configure and
    run a stochastic simulation of a reaction network. It defines the variable
    names, initial states, reaction rates, transition matrix, and propensity
    functions for the simulation. The simulation is executed over a preset
    duration and number of repetitions. The total execution time is printed
    at the end.

    Attributes:
        vars (list of str): Names of the variables/states in the reaction model.
        ini (np.ndarray of int): Initial quantities for each variable/state.
        rates (np.ndarray of float): Reaction rates for the system.
        tm (np.ndarray of int): Transition matrix defining state transitions.
        prop (list of callable): List of propensity functions for the reactions.

    Raises:
        ImportError: If `chisurf.math.reaction._reaction` is not found or cannot 
            be imported.
        RuntimeError: If the model configuration or simulation execution fails.
    """
    import numpy as np
    import time

    import chisurf.math.reaction._reaction as reaction
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


