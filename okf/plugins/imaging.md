---
type: Plugin Group
title: Imaging plugins
description: CLSM/FLIM imaging tools — per-pixel intensity, lifetime, phasor, number-and-brightness and micro-time maps from TTTR imaging data, plus PSF and calibration steps.
resource: chisurf/plugins/microscopy/
tags: [plugins, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

The imaging group reconstructs and analyses confocal (CLSM) TTTR images pixel-by-pixel:
intensity, mean arrival time, phasor-FLIM, lifetime (MLE) and number-and-brightness
maps, with calibration and PSF-determination steps. The plugins live under
`chisurf/plugins/microscopy/`; `imaging_common` holds shared base classes and
`imaging_tools` is the aggregating toolbox.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `microscopy/clsm` | Imaging:CLSM-Draw | Build CLSM-TTTR image representations, pick pixels interactively, export decay histograms. |
| `microscopy/img_pixel_intensity` | Imaging:Intensity | Per-pixel intensity map; writes the standard imaging HDF5 (with source back-reference) that later steps enrich. |
| `microscopy/img_pixel_micro_time` | Imaging:Mean Micro-Time | Per-pixel mean micro-time (arrival-time) maps. |
| `microscopy/img_pixel_phasor` | Imaging:Phasor-FLIM | Per-pixel phasor (g, s) maps and phasor plot. |
| `microscopy/img_pixel_mle` | Imaging:Lifetime:Pixel-wise MLE | Pixel-wise maximum-likelihood lifetime fitting. |
| `microscopy/sm_image_mle` | Imaging:Lifetime:Molecule-wise MLE | Molecule-wise MLE lifetime analysis from PTU imaging data. |
| `microscopy/img_pixel_nb` | Imaging:Number & Brightness | Per-pixel N and B maps. |
| `microscopy/img_calibration` | Imaging:IRF & BG | Per-detector IRF and background calibration, shared with phasor and MLE steps (skippable pipeline step). |
| `microscopy/psf_determination` | Imaging:PSF Determination | 3D Gaussian PSF fitting and bead detection. |
| `microscopy/imaging_tools` | Spectroscopy:Image Tools | Unified toolbox: Image Browser, CLSM Draw, molecule/pixel-wise MLE, PSF Determination. |

These plugins form a pipeline: an HDF5 with a source back-reference is created once
(intensity), then N&B / phasor / MLE steps enrich it, with calibration transferred
between steps. Discovery, activation and declarative panels follow the standard
contract ([plugin system](/architecture/plugin-system.md),
[Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md));
AutoForm's `image`/`waterfall` sections drive the 3D-stack viewers and pixel picking.
The per-pixel CLSM pipeline runs compute in a non-blocking worker process. Correlation
imaging (RICS/ISM) is handled in the acquisition/FCS domain rather than here.
Design context: [PRD-51](/prds/prd-51.md) and [PRD-52](/prds/prd-52.md).
