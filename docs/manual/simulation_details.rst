Simulation details
""""""""""""""""""

The simulations of the β2AR-eGFP-IL3-CT-SNAP measurements (short: CTSNAP) were performed using Burbulator 5 (part of the MFD software package, ).

The NTSNAP construct has both fluorophore on the inner side in a membrane and we assume (i) the fluorophore to be close enough to each other to undergo FRET and (ii) the membrane receptor β2AR to show dynamics such that the fluorophores exchange between two different levels of FRET.

Burbulator uses the Becker&Hickl spc-file format with 4096 TAC channel with a width of 4.07 ps and a laser period of 13.596 ns. The green-to-red detection efficiency ratio and the g-factor was set 1 and the fundamental anisotropy to 0.38. The fluorescence lifetime of both eGFP and SNAP was set to 3 ns with a molecular brightness of 10 kHz/molecule, fluorescence quantum yield of 0.8 and a rotational correlation time of 100 ns. Additionally the background in the green channel was set to 1 kHz and in the red channel to 0.5 kHz. The green crosstalk into the red channels was set to 0.1. All listed values for the fluorophores were adopted based on our measurements from the NTSNAP construct and the setup-describing values were set to reasonable values or 1.

The mean lifetime of the low FRET (LF) and high FRET (HF) states for dynamic exchange was set to 2.4 ns (:emphasis:`E` = 0.2) and 0.9 ns (:emphasis:`E` = 0.7). The equilibrium fractions of LF and HF were set to 0.5 each with a relaxation rate of 71 µs and – in case of triplet – with 16 % triplet blinking at 5.5 µs (Caution: Burbulator adds triplet blinking only to donor molecules!).

The diffusion term was modeled as a bimodal distribution with 30 % of fast diffusing molecules at :emphasis:`tD1` = 1 ms and the rest of the molecules diffusing slowly with t:emphasis:`D2` = 100 ms.

In total, 107 photons were simulated in a 3D Gaussian shaped volume with :emphasis:`w0` = 0.5 µm and :emphasis:`z0` = 1.5 µm, a box size of 20, and :emphasis:`NFCS` = 0.01.

:emphasis:`Of note`: As the slower modeled diffusion time is quite long, the number of photons and the box size might have to be increased further to allow good fitting at large correlation times :emphasis:`tc`. Here, for the sake of time / simplicity, the fitting was stopped at :emphasis:`tc` = 1 sec.
