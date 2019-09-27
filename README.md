[![Codacy Badge](https://api.codacy.com/project/badge/Grade/814238f7b1a14f87821beadabc758408)](https://www.codacy.com/manual/tpeulen/ChiSurf?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Fluorescence-Tools/ChiSurf&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/814238f7b1a14f87821beadabc758408)](https://www.codacy.com/manual/tpeulen/ChiSurf?utm_source=github.com&utm_medium=referral&utm_content=Fluorescence-Tools/ChiSurf&utm_campaign=Badge_Coverage)
[![Linux Build Status](https://travis-ci.org/Fluorescence-Tools/ChiSurf.svg?branch=master)](https://travis-ci.org/Fluorescence-Tools/ChiSurf)

[![Anaconda-Server Version](https://anaconda.org/tpeulen/chisurf/badges/version.svg)](https://anaconda.org/tpeulen/chisurf)
[![DOI](https://zenodo.org/badge/149296509.svg)](https://zenodo.org/badge/latestdoi/149296509)


# ChiSurf

ChiSurf is a software package for the analysis of complex fluorescence data acquired in time-resolved single-molecule 
and ensemble fluorescence experiments. The main feature of ChiSurf is the joint (global) analysis of multiple datasets.
![ChiSurf GUI][1]

## History

The original of ChiSurfs was to estimate errors of model parameters of fluorescence intensity decays in 
Förster-resonance energy transfer (FRET) experiments for FRET-driven structural models of protein and other 
biological macromolecules. It started as a collection of python scripts. With time more features were added, e.g., 
the analysis of correlation curves, correlation of time-tagged-time resolved (TTTR) data. A graphical user interface 
makes these scripts and tools available for users without programming experience.
Overall, ChiSurf is highly experimental and its core is still heavily refactored. Consequently, features, which worked 
in old versions, may be not supported in newer versions, unless users explicitly demand these functions.

## Features

### General features

*  Scripting interface and open API
*  Interactive analysis of multiple datasets
*  Combined analysis of different experimental techniques

### Global analysis

*  Analysis of multiple data sets by joint model function freely defined by the user
*  Freely definable models for FCS analysis & adaptable models for fluorescence decays analysis
*  Fluorescence decay analysis
*  Global analysis of multiple fluorescence decays
*  Generation of fluorescence decay histograms based on TTTR data
*  Analysis of time-resolved anisotropy decays
*  Analysis of FRET quenched fluorescence decays by physical model functions
 
[![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1021/acs.jpcb.7b03441.2222-blue.svg)](http://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b03441)

## Fluorescence correlation spectroscopy

*  Analysis of FCS curves
*  Correlation of TTTR-data by efficient correlation algorithms 
[![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1364/OE.11.003583.2222-blue.svg)](https://doi.org/10.1364/OE.11.003583)

## Simulation of fluorescence observables

*  Kappa2 distributions based on residual anisotropies 
*  Simulation of fluorescence quenching in protein by aromatic amino acids 
[![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1021/acs.jpcb.7b03441.2222-blue.svg)](http://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b03441)
*  Simulation of FRET rate constant distributions based on accessible volumes

# Download

ChiSurf is available as a compiled Windows installation file. By downloading and using ChiSurf, you agree with the 
following terms:

ChiSurf is provided “as is” without warranty of any kind, express or implied. The authors of ChiSurf shall not be, in 
any event, be held liable for any claim, damages or other liability arising from the use of ChiSurf. As the user, you 
are not allowed to redistribute any or all of the code, in any manner to other parties. The downloaded code is for 
your use only. ChiSurf is provided free of charge to both academic and commercial users.

## Windows

*  [19.07.08](https://github.com/Fluorescence-Tools/ChiSurf/releases/download/Stable/chisurf_19.07.09-windows.exe)
*  [17.03.24](https://github.com/Fluorescence-Tools/ChiSurf/releases/download/17.03.24/windows_17.03.24.exe)
*  [Legacy](https://drive.google.com/open?id=1GT8i_ZVnUXCIf_GBhk3TaS-T3DRhWHD2)

## MacOS

*  [19.07.08](https://github.com/Fluorescence-Tools/ChiSurf/releases/download/Stable/chisurf_19.07.08-macos.dmg)
*  [17.03.24](https://github.com/Fluorescence-Tools/ChiSurf/releases/download/17.03.24/macos_17.03.24.zip)

## Linux

Linux users with snap scan simply install ChiSurf using snap.

[![Get it from the Snap Store](https://snapcraft.io/static/images/badges/en/snap-store-white.svg)](https://snapcraft.io/chisurf)

# Support

Please submit feature requests, questions, and bugs as GitHub issues. General questions are addressed in the Google
group linked below.

[Group](https://groups.google.com/d/forum/chisurf-software)

# References

1.  Peulen T, Opanasyuk O, Seidel C. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze 
Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. J Phys Chem B. 2017;121(35):8211-8241.

2.  Wahl M, Gregor I, Patting M, Enderlein J. Fast calculation of fluorescence correlation data with asynchronous 
time-correlated single-photon counting. Opt Express. 2003;11(26):3583-3591.

3.  Sindbert S, Kalinin S, Nguyen H, et al. Accurate distance determination of nucleic acids via Förster resonance 
energy transfer: implications of dye linker length and rigidity. J Am Chem Soc. 2011;133(8):2463-2480.

4.  Kalinin S, Peulen T, Sindbert S, et al. A toolkit and benchmark study for FRET-restrained high-precision structural 
modeling. Nat Methods. 2012;9(12):1218-1225.

[1]: /docs/pictures/chisurf_gui.png "ChiSurf GUI"
