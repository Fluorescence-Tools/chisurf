# ChiSurf
ChiSurf is a software package for the analysis of complex fluorescence data acquired in time-resolved single-molecule and ensemble fluorescence experiments. The main feature of ChiSurf is the joint (global) analysis of multiple datasets.
![ChiSurf GUI][1]
 
 
## History
The original of ChiSurfs was to estimate errors of model parameters of fluorescence intensity decays in Förster-resonance energy transfer (FRET) experiments for FRET-driven structural models of protein and other biological macromolecules. It started as a collection of python scripts. With time more features were added, e.g. the analysis of correlation curves, correlation of time-tagged-time resolved (TTTR) data. A graphical user interface makes these scripts and tools available for users without programming experience.
Overall, ChiSurf is highly experimental and its core is still heavily refactored. Consequently, features, which worked in old versions, may be not supported in newer versions, unless users explicitly demand these functions.

## Features
### General features
* Scripting interface and open API
* Interactive analysis of multiple datasets
* Combined analysis of different experimental techniques

### Global analysis
* Analysis of multiple data sets by joint model function freely defined by the user
* Freely definable models for FCS analysis & adaptable models for fluorescence decays analysis
* Fluorescence decay analysis
* Global analysis of multiple fluorescence decays
* Generation of fluorescence decay histograms based on TTTR data
* Analysis of time-resolved anisotropy decays
* Analysis of FRET quenched fluorescence decays by physical model functions [![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1021/acs.jpcb.7b03441.2222-blue.svg)](http://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b03441)

## Fluorescence correlation spectroscopy
* Analysis of FCS curves
* Correlation of TTTR-data by efficient correlation algorithms [![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1364/OE.11.003583.2222-blue.svg)](https://doi.org/10.1364/OE.11.003583)

## Simulation of fluorescence observables
* Kappa2 distributions based on residual anisotropies 
* Simulation of fluorescence quenching in protein by aromatic amino acids [![DOI for citing FPS](https://img.shields.io/badge/DOI-10.1021/acs.jpcb.7b03441.2222-blue.svg)](http://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b03441)
* Simulation of FRET rate constant distributions based on accessible volumes

# Download
ChiSurf is available as a compiled Windows installation file. By downloading and using ChiSurf, you agree with the following terms:

ChiSurf is provided “as is” without warranty of any kind, express or implied. The authors of ChiSurf shall not be, in any event, be held liable for any claim, damages or other liability arising from the use of ChiSurf. As the user, you are not allowed to redistribute any or all of the code, in any manner to other parties. The downloaded code is for your use only. ChiSurf is provided free of charge to both academic and commercial users.

## Windows

* [17.03.24](https://drive.google.com/open?id=1XJJDW9ESdGqDUhYOj06Lztajn6se3nDe)
* [16.05.14](https://drive.google.com/open?id=1siQgGgRyaaEVNicz5gICjw95WIRtu93U)

## MacOS

* [15.05.14](https://drive.google.com/open?id=18uPP7FmM8-3aYJLx15TjPESNV8SDQfQg) (Tested on macOS 10.12, reported problems on 10.13)


# References
1. Peulen T, Opanasyuk O, Seidel C. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. J Phys Chem B. 2017;121(35):8211-8241. [PubMed]
2. Wahl M, Gregor I, Patting M, Enderlein J. Fast calculation of fluorescence correlation data with asynchronous time-correlated single-photon counting. Opt Express. 2003;11(26):3583-3591. [PubMed]
3. Sindbert S, Kalinin S, Nguyen H, et al. Accurate distance determination of nucleic acids via Förster resonance energy transfer: implications of dye linker length and rigidity. J Am Chem Soc. 2011;133(8):2463-2480. [PubMed]
4. Kalinin S, Peulen T, Sindbert S, et al. A toolkit and benchmark study for FRET-restrained high-precision structural modeling. Nat Methods. 2012;9(12):1218-1225. [PubMed]

[1]: /doc/chisurf_gui.png "ChiSurf GUI"
