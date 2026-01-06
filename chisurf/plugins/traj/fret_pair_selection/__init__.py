"""Optimal FRET pair selection from trajectories (Greedy Olga).

This plugin selects an informative set of FRET pairs from a structural ensemble.
Given a trajectory and a pool of candidate labeling positions (residues), it
computes per-frame FRET efficiencies for all candidate pairs and ranks them by
expected structural precision using the Olga greedy selection algorithm.

The algorithm is based on the experiment planning / informative pair selection
procedure described in:
Dimura, M., Peulen, T.O., Hanke, C.A., Prakash, A., Gohlke, H. and Seidel, C.A.,
2016. Quantitative FRET studies and integrative modeling unravel the structure
and dynamics of biomolecular systems. Current Opinion in Structural Biology,
40, pp.163-185.
"""

name = "Structure:FRET:Optimal Pair Selection"


if __name__ == "plugin":
    from .wizard import FRETPairSelectionWindow

    window = FRETPairSelectionWindow()
    window.show()
