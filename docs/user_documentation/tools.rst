*****
Tools
*****

Time-resolved fluorescence
==========================

Dye-diffusion
-------------

.. figure:: ./figures/dye_diffusion.png
    :align: center
    :figclass: align-center


FRET
====

FRET-lines
----------


.. figure:: ./figures/fret_line_calculator.png
    :figwidth: 50%
    :align: right
    :figclass: align-right



Transfer-calculator
-------------------


.. figure:: ./figures/lifetime_calculator.png
    :align: right
    :figclass: align-right


Anisotropy
==========

kappa2-distributions
--------------------

Structural-Models
=================

Filter-Structures
-----------------

Structure2Transfer
------------------

.. figure:: ./figures/structure2transfer.png
    :figwidth: 10%
    :align: left
    :figclass: align-left


Structure2Transfer is a small tool that calculates for a given set of PDB-files the distance and
the orientation factor kappa2. No fluorophore simulation is performed. Therefore the flurophores
have to be part to the analyzed structures. You can access the tool Structure2Transfer as depicted below.

.. figure:: ./figures/structure2transfer_fig1.png
    :align: right
    :figclass: align-right

After opening the tool the following window becomes visible. Now first open a PDB-file representative
for the whole set. The program does not do any checking if atom numbers within the whole set differ.
Therefore make sure all PDB-files are identical except for their coordinates. After opening the “Reference-structure”
you will be able to define you Donor and Acceptor fluorophore. The atoms chosen in the “Donor” and “Acceptor”
box define the dipole of the donor and the acceptor-fluorophore. If the dipole checkbox is not checked only the
first atom in the donor and acceptor definition box will be used to calculate the distances.

If the “Histogram” checkbox is checked histograms of the orientation factor kappa2, the donor- and acceptor
distance and the FRET-rate will be plotted after the processing of the PDB-files. After processing of the
PDB-files the obtained distances, orientation factors and FRET-rate files can be saved using the “save” button.
The distance between the flurophores and the orientation factor is calculated as depicted above.


To calculate the donor-acceptor distance in every structure, on each fluorophore, we chose two
Cbeta-atoms on the beta-barrel, so that the connecting vector of the two atoms is a good approximation of
the transition dipole. The distance between the middle points of the connecting vectors of the donor and
acceptor is taken as the distance between the chromophores, :math:`R_{DA}`. Table 2 lists out the
detailed calculation steps. For every simulated structure, given the D-A distance and the orientation
factor the FRET-rate was calculated according to :math:`{k_{FRET}} = \left( {{3 \mathord{\left/ {\vphantom {3 2}} \right. \kern-\nulldelimiterspace} 2}} \right) \cdot {\kappa ^2} \cdot \left( {{1 \mathord{\left/{\vphantom {1 {{\tau _0}}}} \right.\kern-\nulldelimiterspace} {{\tau _0}}}} \right) \cdot {\left( {{{{R_0}} \mathord{\left/{\vphantom {{{R_0}} {{R_{DA}}}}} \right.\kern-\nulldelimiterspace} {{R_{DA}}}}} \right)^6}`,
in which :math:`\tau_0` is 2.6ns and the Forster radius, R0, of GFP and mCherry is 52 Ang including :math:`\kappa^2=2/3`.
The steady-state FRET efficiency Ess was determined using :math:`{E_{ss}} = {{{k_{FRET}}} \mathord{\left/ {\vphantom {{{k_{FRET}}} {\left( {{k_{FRET}} + {k_0}} \right)}}} \right. \kern-\nulldelimiterspace} {\left( {{k_{FRET}} + {k_0}} \right)}}`.


.. figure:: ./figures/structure2transfer_fig2.png
    :figwidth: 40%
    :align: center
    :figclass: align-center



Trajectory-converter
--------------------

Potential-calculation
---------------------

Labeling-file generation
------------------------

