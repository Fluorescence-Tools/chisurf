*********
Tutorials
*********


TCSPC
=====

Gaussian
********

Under our measurement conditions we have a mixture of monomeric and dimeric
hGBP1. Thus all donor fluorescence decays FD(t) were fitted with the decays
of at least two molecular species, FD(0)(t) for the Donor-only species
and FD(A)(t) for the dimeric FRET species:

.. math::
    `F(t) = (1 - {x_{{\rm{DOnly}}}}){F_{{\rm{D(A)}}}}(t) + {x_{{\rm{DOnly}}}}{F_{{\rm{D(0)}}}}(t) + c`

where xDOnly corresponds to the fraction of Donor-only molecules and c is a
constant offset. Due to local quenching the fluorescence decay of the donor
in the absence of FRET is tri-exponential with the individual species fractions
:math:`x_{\rm{D}}^{(i)}` and fluorescence lifetimes :math:`tau _{{\rm{D(0)}}}^{(i)}`
(see Table S3)

Thus, the time resolved fluorescence intensity decays of donor-/acceptor-labeled
protein-complex (FRET-sample) were fitted globally with the decays of the
donor-/unlabeled protein-complexes (donor only sample, DOnly). Generally it is
reasonable to assume that the radiative lifetime of the donor is not affected
by quenching. Hence, the FRET-rate constant (kFRET) is only determined by the
donor-acceptor distance and their relative orientation
[Ref: Kalinin, S.,& Johansson, L.B.-Å. Energy Migration and Transfer Rates Are
Invariant to Modeling the Fluorescence Relaxation by Discrete and Continuous
Distributions of Lifetimes. (2004) J. Phys. Chem. B 108, 3092-3097].
Expressing the FRET-rate constant in terms of distances the donor-fluorescence
in presence of acceptor is given by:

.. math::
    `{F_{{\rm{D}}(0)}}(t) = \sum\limits_i {x_{\rm{D}}^{(i)}\exp ( - t/\tau _{{\rm{D}}(0)}^{(i)})}`

Whereas p(RDA) is a FRET-rate distribution expressed as distance and R0 is the
Forster-radius (in this case R0 =52 Å) and :math:`k_0=1/\tau_0` is the radiative
rate the unquenched dye. The flurophores are attached to the biomolecule by long
flexible linkers. Hence, a donor-acceptor distance distribution is expected
which is not averaged during the fluorescence lifetime of the dyes
[Sindbert, S., Kalinin, S., Nguyen, H., Kienzler, A., Clima, L., Bannwarth, W., Appel, B., Müller, S. &
Seidel, C.A.M. (2011) J. Am. Chem. Soc. 33, 2463-2480.] and the fluorescence
decay FD(A) has to be expressed as by an donor-acceptor distance distribution
p(RDA) with a non-zero width. Here the experimental time-resolved fluorescence
intensities were either fitted by a Gaussian distribution of donor-acceptor
distances (p(RDA)) with a mean inter dye distance RDA and a width wDA (Eq. 4)
or, analog to the Tikhonov regularization, p(RDA) was determined by
deconvolution of the fluorescence intensity decays by using the maximum-entropy
method (MEM) [Livesey, A. K.; Skilling J. Maximum Entropy Theory. Acta Crystallogr. Sect. A 1985, 41, 113-122. Ref. Brochon, J. C. Methods Enzymol.
1994, 240, 262-311.].

.. math::
    `{F_{{\rm{D(A)}}}}(t) = {F_{D(0)}} \cdot \int\limits_{{R_{{\rm{DA}}}}} {p({R_{{\rm{DA}}}}) \cdot \exp \left( { - t \cdot {k_0} \cdot {{({R_0}/{R_{{\rm{DA}}}})}^6}} \right)\,d{R_{{\rm{DA}}}}}`

The width of the Gaussian donor-acceptor distance distribution wDA should not be misinterpreted as the
experimental/statistical-error but it describes a real physical property of the donor-acceptor pair. The
experimental fluorescence decays presented below are described by combining the above formulas and were fitted
by custom software written in Python.

:math:`{F_{{\rm{D(A)}}}}(t) = {F_{D(0)}} \cdot \int\limits_{{R_{{\rm{DA}}}}} {\frac{1}{{{w_{{\rm{DA}}}}\sqrt {{\pi  \mathord{\left/ {\vphantom {\pi  2}} \right. \kern-\nulldelimiterspace} 2}} \,}}\exp \left( { - 2\,{{\left[ {\frac{{{R_{{\rm{DA}}}} - \left\langle {{R_{{\rm{DA}}}}} \right\rangle }}{{{w_{DA}}}}} \right]}^2}} \right)\exp \left( { - t \cdot {k_0}\left[ {1 + {{({R_0}/{R_{{\rm{DA}}}})}^6}} \right]} \right)\,d{R_{{\rm{DA}}}}}`



Stopped Flow
============

In the reaction system model a system of reactions has to be defined by the user. Given the user-defined reaction
system and initial values of the reacting species the time-evolution of the species is calculated by numerical
integration of the differential equations. Below the theoretical frame-work is described followed by a description
how to define custom reaction models and fit experimental data.
Given a set of species   and a number of   elementary reactions with associated rate constants  the reaction system
can be described by the following chemical reactions:

.. math::
    \begin{array}{*{20}{c}}
      {{e_{11}}{X_1} + {e_{21}}{X_2} +  \ldots  + {e_{N1}}{X_N}}&{\xrightarrow{{k1}}}&{{p_{11}}{X_1} + {p_{21}}{X_2} +  \ldots  + {p_{N1}}{X_N}} \\
      \begin{gathered}
       \vdots  \\
      {e_{1j}}{X_1} + {e_{2j}}{X_2} +  \ldots  + {e_{Nj}}{X_N} \\
       \vdots  \\
    \end{gathered} &\begin{gathered}
       \vdots  \\
      \xrightarrow{{kj}} \\
       \vdots  \\
    \end{gathered} &\begin{gathered}
       \vdots  \\
      {p_{1j}}{X_1} + {p_{2j}}{X_2} +  \ldots  + {p_{Nj}}{X_N} \\
       \vdots  \\
    \end{gathered}  \\
      {{e_{1R}}{X_1} + {e_{2R}}{X_2} +  \ldots  + {e_{NR}}{X_N}}&{\xrightarrow{{kR}}}&{{p_{1R}}{X_1} + {p_{2R}}{X_2} +  \ldots  + {p_{NR}}{X_N}}
    \end{array}

Using the law of mass action the flux of molecules per time and unit-volume is given by:

.. math::
    {f_j}\left( {\vec c} \right) = {k_j}\prod\limits_{i = 1}^N {c{{({X_i})}^{{e_i}}}}

.. math::
    \frac{{dc\left( {{X_i}} \right)}}{{dt}} = \sum\limits_{j = 1}^R {{p_{ij}}{f_j}\left( {\vec c} \right) - {e_{ij}}{f_j}\left( {\vec c} \right)}

.. math::
    I(t) = s \cdot \left\langle {\vec c(t),\vec q} \right\rangle

.. math::
    I'(t) = \frac{{\int {D(t)dt} }}{{\int {I(t)dt} }} \cdot I(t)

.. math::
    \begin{array}{*{20}{c}}
      A&{\xrightarrow{{kf}}}&B \\
      B&{\xrightarrow{{kb}}}&A \\
      B&{\xrightarrow{{kR}}}&C
    \end{array}