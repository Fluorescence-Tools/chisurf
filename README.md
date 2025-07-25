![conda build](https://github.com/fluorescence-tools/chisurf/actions/workflows/conda-build.yml/badge.svg)
[![Anaconda-Server Version](https://anaconda.org/tpeulen/chisurf/badges/version.svg)](https://anaconda.org/tpeulen/chisurf)
[![DOI](https://zenodo.org/badge/149296509.svg)](https://zenodo.org/badge/latestdoi/149296509)

# ChiSurf

ChiSurf is a software package for the global analysis of fluorescence data. It enables users to interlink, optimize, and jointly sample variables of models for time-resolved single-molecule and ensemble fluorescence experiments. By introducing dependencies across models, ChiSurf allows for the construction of complex descriptions across multiple datasets.
For a detailed explanation of the methods and implementation, please refer to the [ChiSurf Preprint](https://www.preprints.org/manuscript/202501.1615/v1).

<p align="center">
  <img src="https://www.peulen.xyz/wp-content/uploads/2017/05/ChiSurf_Overview-768x451.png" alt="ChiSurf GUI" width="50%">
  <img src="https://www.peulen.xyz/wp-content/uploads/2024/04/ChiSurf_ParameterGraph.png" alt="ChiSurf Parameter Network" width="40%">
</p>

## Features

### General Features

- **Scripting Interface & Open API:** Flexible integration into existing workflows.
- **Interactive Analysis:** Simultaneously analyze multiple datasets.
- **Combined Analysis:** Joint analysis of different experimental techniques.

### Global Analysis

- Analysis of multiple datasets with user-defined joint model functions.
- Freely definable models for fluorescence correlation spectroscopy (FCS) and fluorescence decay analysis.
- Global analysis of multiple fluorescence decays.
- Generation of fluorescence decay histograms based on TTTR data.
- Analysis of time-resolved anisotropy decays.
- FRET-quenched fluorescence decay analysis using physical model functions.

### Fluorescence Correlation Spectroscopy (FCS)

- Analysis of FCS curves.
- Efficient correlation algorithms for TTTR data.

### Simulation of Fluorescence Observables

- Simulation of kappa² distributions based on residual anisotropies.
- Simulation of fluorescence quenching in proteins by aromatic amino acids.
- Simulation of FRET rate constant distributions based on accessible volumes.

---

## Download

ChiSurf is available as a compiled Windows installation file. By downloading and using ChiSurf, you agree to the following terms:

> ChiSurf is provided “as is” without warranty of any kind, express or implied. The authors of ChiSurf shall not be held liable for any claim, damages, or other liability arising from its use. Redistribution of the code is not permitted, and it is provided free of charge for both academic and commercial users.

### Installation Instructions

#### Linux & macOS

Install `chisurf` using `conda` or `mamba`:

```bash
conda create -n chisurf chisurf -c conda-forge -c tpeulen
conda activate chisurf
```

3. Launch ChiSurf GUI:

```bash
chisurf
```

#### Windows

Download the latest releases from [https://www.peulen.xyz/downloads/](https://www.peulen.xyz/downloads/).

---

## Tutorials

Learn how to use ChiSurf through the following video tutorials:

### General Introduction  
[![General Introduction](https://img.youtube.com/vi/qa4UQnhO-8M/0.jpg)](https://www.youtube.com/watch?v=qa4UQnhO-8M)

### Fluorescence Decay Analysis  
[![Fluorescence Decay Analysis](https://img.youtube.com/vi/rtllur-jUag/0.jpg)](https://www.youtube.com/watch?v=rtllur-jUag)

### Fluorescence Decay Analysis  
[![Fluorescence Decay Analysis](https://img.youtube.com/vi/rtllur-jUag/0.jpg)](https://www.youtube.com/watch?v=rtllur-jUag)

### Fluorescence Correlation Spectroscopy (FCS)  
[![FCS](https://img.youtube.com/vi/k9NgYbyLyXk/0.jpg)](https://www.youtube.com/watch?v=k9NgYbyLyXk)


## Support

Please submit feature requests, questions, and bugs as GitHub issues. General questions are addressed and discussed in
the Discord [group](https://discord.gg/mFEDHURSnJ)

## References

1. Peulen T, Opanasyuk O, Seidel C. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze
Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. J Phys Chem B. 2017;121(35):8211-8241.

2. Wahl M, Gregor I, Patting M, Enderlein J. Fast calculation of fluorescence correlation data with asynchronous
time-correlated single-photon counting. Opt Express. 2003;11(26):3583-3591.

3. Sindbert S, Kalinin S, Nguyen H, et al. Accurate distance determination of nucleic acids via Förster resonance
energy transfer: implications of dye linker length and rigidity. J Am Chem Soc. 2011;133(8):2463-2480.

4. Kalinin S, Peulen T, Sindbert S, et al. A toolkit and benchmark study for FRET-restrained high-precision structural
modeling. Nat Methods. 2012;9(12):1218-1225.

[1]: https://www.peulen.xyz/wp-content/uploads/2017/05/ChiSurf_Overview-768x451.png "ChiSurf GUI"
[2]: https://www.peulen.xyz/wp-content/uploads/2024/04/ChiSurf_ParameterGraph.png "ChiSurf Parameter Network"
